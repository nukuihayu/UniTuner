"""Stage ComfyUI denoiser layouts a block at a time before precision conversion."""

import json
import re
import tempfile
from collections.abc import Generator, Iterator
from contextlib import ExitStack, contextmanager
from importlib import import_module
from pathlib import Path

import torch
from torch import Tensor

from unitunner.artifacts.formats.precision_policy import TEXT_FAMILIES, resolve_family, unprefixed
from unitunner.artifacts.formats.quantization import FORMATS
from unitunner.artifacts.formats.transformer import (
    convert_transformer,
    dequantize_transformer,
    is_diffusers_transformer,
)
from unitunner.artifacts.quantization import json_tensor
from unitunner.artifacts.tensors import TensorReader, resolve_shards, save_tensor_stream
from unitunner.models.components.transformer_artifacts import published_config

TRANSFORMER_TYPES = {
    "z-image": ("z_image", "ZImage"),
    "moss-image": ("moss_image", "MossImage"),
    "flux": ("flux", "Flux"),
    "flux2": ("flux2", "Flux2"),
    "flux3": ("flux3", "Flux3"),
    "qwen-image": ("qwen_image", "QwenImage"),
    "qwen-image-2-1": ("qwen_image_2_1", "QwenImage21"),
    "wan": ("wan", "Wan"),
    "krea2": ("krea2", "Krea2"),
    "minimax-h3": ("minimax_h3", "MiniMaxH3"),
}


def _z_image_comfyui(tensors: dict[str, Tensor]) -> Iterator[tuple[str, Tensor]]:
    tensors = dequantize_transformer(tensors)
    for key in tuple(tensors):
        if key.endswith(".attention.to_q.weight"):
            keys = [key.replace(".to_q.", f".to_{part}.") for part in ("q", "k", "v")]
            if any(name not in tensors for name in keys):
                raise ValueError(f"Incomplete Z-Image QKV projections: {key}")
            tensors[key.replace(".to_q.", ".qkv.")] = torch.cat([tensors.pop(name) for name in keys], dim=0)
    for key, value in tensors.items():
        for old, new in (("all_x_embedder.2-1.", "x_embedder."), ("all_final_layer.2-1.", "final_layer.")):
            if key.startswith(old):
                key = new + key[len(old) :]
                break
        key = key.replace(".attention.to_out.0.", ".attention.out.")
        key = key.replace(".attention.norm_q.", ".attention.q_norm.").replace(
            ".attention.norm_k.", ".attention.k_norm."
        )
        yield key, value


@contextmanager
def comfyui_source(source: Path, *, family: str, precision: str) -> Generator[tuple[Path, str]]:
    """Keep existing consumer layouts; stage layout changes before quantization.

    No complete model is allocated. Only one transformer block's tensors and
    fused projections are resident at a time. Temporary files are cleaned after
    conversion, and the destination is still published atomically by its owner.
    """
    with ExitStack() as stack:
        readers: dict[str, TensorReader] = {}
        metadata: dict[str, str] = {}
        for shard in resolve_shards(source):
            reader = stack.enter_context(TensorReader(shard))
            for key, value in reader.metadata().items():
                if key in metadata and metadata[key] != value:
                    raise ValueError(f"Conflicting shard metadata: {key}")
                metadata[key] = value
            for key in reader.keys():
                if key in readers:
                    raise ValueError(f"Duplicate tensor key across shards: {key}")
                readers[key] = reader
        # Ordinary casting retains generic/text namespaces without model guesses.
        if family == "auto" and precision not in FORMATS:
            try:
                resolved = resolve_family(family, metadata, readers)
            except ValueError as error:
                if "Ambiguous model layout" not in str(error):
                    raise
                from unitunner.logging import get_logger

                get_logger("artifacts").debug("Using generic layout for ordinary casting: %s", error)
                resolved = "generic"
        else:
            resolved = resolve_family(family, metadata, readers)
        if resolved == "generic" or resolved in TEXT_FAMILIES:
            yield source, resolved
            return
        names = {unprefixed(key) for key in readers}
        module, stem = TRANSFORMER_TYPES[resolved]
        external = is_diffusers_transformer(module, names)
        z_split = resolved == "z-image" and any(".attention.to_q." in key for key in names)
        serialized = metadata.get(f"unitunner.{module}.config")
        if resolved == "qwen-image" and serialized is None:
            from unitunner.models.qwen_image.config.model import QwenImageConfig

            serialized = published_config(source, QwenImageConfig).model_dump_json()
            metadata[f"unitunner.{module}.config"] = serialized
        edit_marker = (
            resolved == "qwen-image"
            and serialized is not None
            and json.loads(serialized).get("variant") == "edit-2511"
            and "__index_timestep_zero__" not in names
        )
        if not external and not z_split and not edit_marker:
            yield source, resolved
            return
        if precision == "keep" and any(key.endswith(".comfy_quant") for key in names) and (external or z_split):
            raise ValueError("Changing packed transformer layout requires an explicit output precision")
        expected: dict[str, Tensor] = {}
        if external:
            cls = getattr(import_module(f"unitunner.models.{module}.config.model"), stem + "Config")
            config = cls.model_validate_json(serialized) if serialized is not None else published_config(source, cls)
            constructor = getattr(import_module(f"unitunner.models.{module}.transformer"), stem + "Transformer")
            with torch.device("meta"):
                expected = constructor(config).state_dict()
            metadata[f"unitunner.{module}.config"] = config.model_dump_json()
        prefix = next(
            (
                prefix
                for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer.")
                if readers and all(key.startswith(prefix) for key in readers)
            ),
            "",
        )
        legacy = {
            name + ".comfy_quant": json_tensor(config)
            for name, config in json.loads(metadata.get("_quantization_metadata", "{}")).get("layers", {}).items()
        }
        if legacy.keys() & readers.keys():
            raise ValueError("Duplicate ComfyUI quantization descriptions")
        groups: dict[str, list[str]] = {}
        for key in (*readers, *legacy):
            name = key[len(prefix) :]
            match = re.match(
                r"((?:layers|blocks|double_blocks|single_blocks|transformer_blocks|single_transformer_blocks|noise_refiner|context_refiner|txt_mode_blocks)\.\d+|(?:content_mode_blocks|mode_blocks)\.[^.]+\.\d+|(?:token_refiner\.(?:blocks|refiner_blocks)|(?:text_fusion|txtfusion)\.(?:layerwise_blocks|refiner_blocks))\.\d+)\.",
                name,
            )
            groups.setdefault(match[1] if match else "__roots__", []).append(key)

        def stream():
            seen: set[str] = set()
            for keys in groups.values():
                tensors = {key[len(prefix) :]: legacy[key] if key in legacy else readers[key].read(key) for key in keys}
                native = convert_transformer(module, tensors, expected, external=True) if external else tensors
                if external:
                    for key, value in native.items():
                        if key not in expected or value.shape != expected[key].shape or not value.is_floating_point():
                            raise ValueError(f"Invalid external transformer key/shape: {key}")
                    seen.update(native)
                if z_split:
                    yield from _z_image_comfyui(native)
                else:
                    yield from native.items()
                del tensors, native
            if external and seen != expected.keys():
                raise ValueError(f"Missing external transformer keys: {sorted(expected.keys() - seen)}")
            if edit_marker:
                yield "__index_timestep_zero__", torch.empty(0)

        metadata.pop("_quantization_metadata", None)
        temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix="unitunner-comfyui-"))
        staged = Path(temporary) / "model.safetensors"
        save_tensor_stream(stream(), staged, metadata=metadata)
        yield staged, resolved
