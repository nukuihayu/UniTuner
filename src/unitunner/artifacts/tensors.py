"""Read and stream safetensors without model, execution or quantization dependencies."""

import json
import re
import shutil
import struct
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Self

import torch
from safetensors import safe_open
from torch import Tensor

from unitunner.artifacts.atomic import atomic_output
from unitunner.artifacts.hashing import canonical_sha256, file_sha256

_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
    torch.float8_e5m2: "F8_E5M2",
    torch.float8_e4m3fn: "F8_E4M3",
}
_MAX_HEADER_BYTES = 100_000_000
_SHARD = re.compile(r"^(.*?)([0-9]+)-of-([0-9]+)\.safetensors$")


def save_tensor_stream(
    tensors: Iterable[tuple[str, Tensor]], path: Path, *, metadata: Mapping[str, str] | None = None
) -> None:
    """Atomically save a single-pass stream with memory bounded by one tensor.

    Spool payloads to disk while collecting the header; copy the payload after
    writing the completed header. Peak temporary disk space is twice the output
    size. A duplicate key, unsupported tensor or failed generator preserves the
    previous destination. No source dtype, tensor or metadata is mutated.
    """
    if metadata is not None and any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ValueError("Safetensors metadata requires string keys and values")
    header: dict[str, dict[str, str] | dict[str, str | list[int]]] = (
        {} if metadata is None else {"__metadata__": dict(metadata)}
    )
    offset = 0
    with atomic_output(path) as temporary, tempfile.TemporaryFile(dir=path.parent) as payload:
        for key, tensor in tensors:
            if not isinstance(key, str) or key == "__metadata__" or key in header:
                raise ValueError(f"Invalid or duplicate tensor key: {key!r}")
            if tensor.layout != torch.strided or tensor.is_meta or tensor.dtype not in _DTYPE_NAMES:
                raise ValueError(f"Unsupported tensor layout, storage or dtype for {key!r}")
            size = tensor.numel() * tensor.element_size()
            header[key] = {
                "dtype": _DTYPE_NAMES[tensor.dtype],
                "shape": list(tensor.shape),
                "data_offsets": [offset, offset + size],
            }
            offset += size
            raw = tensor.detach().resolve_conj().resolve_neg().to("cpu").reshape(-1).contiguous().view(torch.uint8)
            payload.write(memoryview(raw.numpy()))
            del raw, tensor
        encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        if len(encoded) > _MAX_HEADER_BYTES:
            raise ValueError("Safetensors header exceeds the format limit")
        payload.seek(0)
        with temporary.open("wb") as stream:
            stream.write(struct.pack("<Q", len(encoded)))
            stream.write(encoded)
            shutil.copyfileobj(payload, stream, length=8 * 1024 * 1024)


def save_tensors(
    tensors: Mapping[str, Tensor],
    path: Path,
    *,
    metadata: Mapping[str, str] | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    """Atomically publish a file while staging at most one tensor on CPU.

    Keys and shapes are retained; logical views are serialized contiguously.
    With dtype=None, retain every dtype. Explicit FP32/FP16/BF16 casts affect
    floating tensors only, using PyTorch rounding; integer/bool tensors retain
    their dtype and exact values. Header offsets use the serialized element size,
    including scalars and empty tensors. Casting occurs during per-tensor CPU
    staging, without modifying live parameters, gradients or optimizer state.
    Reduced precision can lose range and mantissa bits. Metadata is
    string-only. Concurrent caller mutation is unsupported. A write failure
    leaves the previous destination intact and removes the temporary file.
    """
    if dtype is not None and dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Safetensors floating export dtype requires FP32, FP16 or BF16")
    cast_size = 0 if dtype is None else torch.finfo(dtype).bits // 8
    entries = tuple(tensors.items())
    quantized = {
        key.removesuffix("comfy_quant") for key in tensors if key == "comfy_quant" or key.endswith(".comfy_quant")
    }
    cast_keys = {
        key
        for key, tensor in entries
        if dtype is not None
        and tensor.is_floating_point()
        and not any(
            key.startswith(prefix + "weight_") or key in (prefix + "input_scale", prefix + "pre_quant_scale")
            for prefix in quantized
        )
    }
    if metadata is not None and any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ValueError("Safetensors metadata requires string keys and values")
    header: dict[str, dict[str, str] | dict[str, str | list[int]]] = {}
    if metadata is not None:
        header["__metadata__"] = dict(metadata)
    offset = 0
    for key, tensor in entries:
        if not isinstance(key, str) or key == "__metadata__":
            raise ValueError("Tensor keys must be strings other than '__metadata__'")
        if tensor.layout != torch.strided or tensor.is_meta or tensor.dtype not in _DTYPE_NAMES:
            raise ValueError(f"Unsupported tensor layout, storage or dtype for {key!r}")
        target_dtype = dtype if key in cast_keys and dtype is not None else tensor.dtype
        size = tensor.numel() * (tensor.element_size() if target_dtype == tensor.dtype else cast_size)
        header[key] = {
            "dtype": _DTYPE_NAMES[target_dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    if len(encoded) > _MAX_HEADER_BYTES:
        raise ValueError("Safetensors header exceeds the format limit")
    with atomic_output(path) as temporary, temporary.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for key, tensor in entries:
            if tensor.numel():
                raw = (
                    tensor.detach()
                    .resolve_conj()
                    .resolve_neg()
                    .to(device="cpu", dtype=dtype if key in cast_keys else tensor.dtype)
                    .reshape(-1)
                    .contiguous()
                    .view(torch.uint8)
                )
                stream.write(memoryview(raw.numpy()))
                del raw


class TensorReader(AbstractContextManager["TensorReader"]):
    """Own a validated Rust reader; CPU tensors share mmap storage and outlive close.

    Explicit device copies are blocking. No global device synchronization or
    model-specific conversion occurs. Create one context per worker.
    """

    def __init__(self, path: Path) -> None:
        self._reader: safe_open = safe_open(path, framework="pt", device="cpu")
        self._reader.__enter__()
        self._closed: bool = False

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("TensorReader is closed")

    def close(self) -> None:
        """Release this reader's mapping reference exactly once."""
        if not self._closed:
            self._closed = True
            self._reader.__exit__(None, None, None)

    def keys(self) -> tuple[str, ...]:
        """Return deterministic tensor keys, excluding metadata."""
        self._ensure_open()
        return tuple(sorted(self._reader.keys()))

    def metadata(self) -> dict[str, str]:
        """Return a detached string-only metadata mapping."""
        self._ensure_open()
        return dict(self._reader.metadata() or {})

    def shape(self, key: str) -> tuple[int, ...]:
        """Read a tensor's logical dimensions from its header without touching payload pages."""
        self._ensure_open()
        return tuple(self._reader.get_slice(key).get_shape())

    def dtype(self, key: str) -> str:
        """Read the safetensors dtype tag without materializing tensor payloads."""
        self._ensure_open()
        return self._reader.get_slice(key).get_dtype()

    def read(self, key: str, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """Materialize an exact tensor, converting only explicit dtype/device choices."""
        self._ensure_open()
        tensor = self._reader.get_tensor(key)
        return tensor if device is None and dtype is None else tensor.to(device=device, dtype=dtype)


def resolve_shards(path: Path) -> tuple[Path, ...]:
    """Resolve shards using actual numeric field widths; reject missing files early."""
    path = resolve_weight_path(path)
    if path.name.endswith(".index.json"):
        index = json.loads(path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(not isinstance(key, str) or not isinstance(value, str) for key, value in weight_map.items())
        ):
            raise ValueError(f"Invalid safetensors weight index: {path}")
        names = set(weight_map.values())
        if any(
            not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors") for name in names
        ):
            raise ValueError("Weight index must reference sibling safetensors files")
        paths = tuple(path.parent / name for name in sorted(names))
        actual: dict[str, str] = {}
        for shard in paths:
            with TensorReader(shard) as reader:
                for key in reader.keys():
                    if key in actual:
                        raise ValueError(f"Duplicate indexed tensor: {key}")
                    actual[key] = shard.name
        if actual != weight_map:
            raise ValueError("Safetensors index differs from shard contents")
        return paths
    match = _SHARD.fullmatch(path.name)
    if match is None:
        paths = (path,)
    else:
        prefix, index_text, count_text = match.groups()
        index, count = int(index_text), int(count_text)
        if not 1 <= index <= count:
            raise ValueError(f"Invalid shard numbering: {path.name}")
        paths = tuple(
            path.with_name(f"{prefix}{i:0{len(index_text)}d}-of-{count_text}.safetensors") for i in range(1, count + 1)
        )
    for shard in paths:
        if not shard.is_file():
            raise FileNotFoundError(shard)
    return paths


def resolve_weight_path(path: Path) -> Path:
    """Select a Diffusers transformer component or an unambiguous weight artifact."""
    if not path.is_dir():
        return path
    if (path / "transformer").is_dir():
        path = path / "transformer"
    indices = sorted(path.glob("*.safetensors.index.json"))
    if len(indices) == 1:
        return indices[0]
    files = sorted(path.glob("*.safetensors")) if not indices else indices
    if len(files) == 1:
        return files[0]
    if files and not indices and all(_SHARD.fullmatch(file.name) for file in files):
        if set(resolve_shards(files[0])) == set(files):
            return files[0]
    raise ValueError(f"Select one weight file or component directory; ambiguous or absent weights in {path}")


def weight_artifact_sha256(path: Path, *, release_cache: bool = False) -> str:
    """Hash all resolved weight shards, retaining single-file digest semantics.

    A numbered shard selects the complete ordered set. Hash a versioned list of
    each shard's bounded-memory content digest, independently of directory names
    or which shard was supplied. Missing shards fail before hashing. A plain file
    returns its existing SHA256, preserving single-file cache/resume identities.
    """
    shards = resolve_shards(path)
    if len(shards) == 1:
        return file_sha256(shards[0], release_cache=release_cache)
    return canonical_sha256(
        {"schema": "weight-shards/1", "sha256": [file_sha256(shard, release_cache=release_cache) for shard in shards]}
    )


def iter_tensors(path: Path) -> Iterator[tuple[str, Tensor]]:
    """Yield one tensor at a time after checking all shard headers for duplicate keys.

    No earlier tensors are retained. The caller owns tensor lifetimes. Files
    must remain immutable throughout preflight and iteration.
    """
    shards = resolve_shards(path)
    seen: set[str] = set()
    for shard in shards:
        with TensorReader(shard) as reader:
            keys = set(reader.keys())
            duplicates = seen & keys
            if duplicates:
                raise ValueError(f"Duplicate tensor keys across shards: {sorted(duplicates)}")
            seen.update(keys)
    for shard in shards:
        with TensorReader(shard) as reader:
            for key in reader.keys():
                yield key, reader.read(key)
