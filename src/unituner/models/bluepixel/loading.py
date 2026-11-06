"""Load BluePixel weights with the native FLUX.2 projection layout."""

from collections.abc import Mapping
from pathlib import Path

import torch

from unituner.artifacts.formats.transformer import convert_transformer
from unituner.artifacts.tensors import save_tensors
from unituner.models.bluepixel.transformer import BluePixelTransformer
from unituner.models.components.state_loading import load_model_state
from unituner.models.components.transformer_artifacts import resolve_transformer_config, transformer_tensors
from unituner.models.flux2.config.model import Flux2Config

_CONFIG_KEY = "unituner.bluepixel.config"
_CPU = torch.device("cpu")


def load_bluepixel(
    path: Path, *, config: Flux2Config | None = None, device: torch.device = _CPU,
    dtype: torch.dtype = torch.bfloat16,
) -> BluePixelTransformer:
    """Validate all native/Diffusers keys and published Klein-9B geometry before assigning weights."""
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("BluePixel requires FP32, FP16 or BF16 weights")
    topology = resolve_transformer_config(
        path, Flux2Config, metadata_key=_CONFIG_KEY, config=config,
        variant="base", version="klein-9b", mismatch_message="BluePixel topology differs from artifact metadata",
    )
    if topology.version != "klein-9b" or topology.variant != "base" or topology.use_guidance_embed:
        raise ValueError("BluePixel requires a Klein-9B backbone without embedded guidance")
    with torch.device("meta"):
        model = BluePixelTransformer(topology)
    tensors = convert_transformer("bluepixel", transformer_tensors(path), model.state_dict())
    load_model_state(model, tensors, device=device, dtype=dtype, label="BluePixel")
    return model


def save_bluepixel(
    model: BluePixelTransformer, path: Path, *, metadata: Mapping[str, str] | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    """Publish native model weights with explicit BluePixel topology metadata."""
    save_tensors(model.state_dict(), path, metadata={**(metadata or {}), _CONFIG_KEY: model.config.model_dump_json()},
                 dtype=dtype)
