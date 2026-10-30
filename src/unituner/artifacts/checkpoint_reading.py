"""Read tensor-only checkpoints without coupling file formats to model topology."""

from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor

from unituner.artifacts.tensors import iter_tensors


def is_pytorch_checkpoint(path: Path) -> bool:
    """Recognize PyTorch extensions and renamed ZIP/legacy pickle files."""
    if path.is_dir():
        return False
    with path.open("rb") as stream:
        signature = stream.read(9)
    # A safetensors header length can itself start with pickle's 0x80 byte.
    if (
        len(signature) == 9
        and signature[8:9] == b"{"
        and 0 < int.from_bytes(signature[:8], "little") < path.stat().st_size
    ):
        return False
    return path.suffix.lower() in (".pt", ".pth", ".bin") or signature.startswith((b"PK\x03\x04", b"\x80"))


def checkpoint_tensors(
    path: Path, *, label: str = "Checkpoint", memory_map: bool = False
) -> Iterator[tuple[str, Tensor]]:
    """Read safe CPU tensors, optionally mapping ZIP storage for accelerator loading."""
    if is_pytorch_checkpoint(path):
        with path.open("rb") as stream:
            mapped = memory_map and stream.read(4) == b"PK\x03\x04"
        # Legacy pickle serialization cannot be mapped. Select by content so
        # renamed official DINO/VGG checkpoints keep working.
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=mapped)
        if not isinstance(state, dict) or any(
            not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()
        ):
            raise ValueError(f"{label} PyTorch checkpoint must contain a tensor state dict")
        yield from state.items()
    else:
        yield from iter_tensors(path)
