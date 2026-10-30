"""Load native FLUX3 safetensors with explicit topology and complete validation."""

from collections.abc import Mapping
from pathlib import Path

import torch

from unituner.artifacts.formats.transformer import convert_transformer
from unituner.artifacts.tensors import save_tensors
from unituner.models.components.state_loading import load_model_state
from unituner.models.components.transformer_artifacts import (
    resolve_transformer_config,
    transformer_tensors,
)
from unituner.models.flux3.config.model import Flux3Config
from unituner.models.flux3.transformer import Flux3Transformer

_CPU = torch.device("cpu")
_CONFIG_KEY = "unituner.flux3.config"


def save_flux3(
    model: Flux3Transformer, path: Path, *, metadata: Mapping[str, str] | None = None, dtype: torch.dtype | None = None
) -> None:
    """Atomically save native keys/topology without modifying live parameters or RNG.

    Values are staged one tensor at a time. No external prefix or fused projection
    conversion is inferred; image/reference/text native keys remain distinct.
    Optional dtype casts only published floating tensors; live weights stay intact.
    """
    save_tensors(
        model.state_dict(),
        path,
        dtype=dtype,
        metadata={**(metadata or {}), _CONFIG_KEY: model.config.model_dump_json()},
    )


def load_flux3(
    path: Path,
    *,
    config: Flux3Config | None = None,
    variant: str | None = None,
    version: str | None = None,
    device: torch.device = _CPU,
    dtype: torch.dtype = torch.float32,
) -> Flux3Transformer:
    """Load native, Diffusers or ComfyUI weights with validated topology, keys and shapes."""
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("FLUX3 native loading supports FP32, FP16 or BF16")
    config = resolve_transformer_config(
        path,
        Flux3Config,
        metadata_key=_CONFIG_KEY,
        config=config,
        variant=variant,
        version=version,
        mismatch_message="Explicit FLUX3 topology differs from artifact metadata",
    )
    with torch.device("meta"):
        model = Flux3Transformer(config)
    expected, tensors = model.state_dict(), transformer_tensors(path)
    tensors = convert_transformer("flux3", tensors, expected)
    load_model_state(model, tensors, device=device, dtype=dtype, label="FLUX3")
    return model
