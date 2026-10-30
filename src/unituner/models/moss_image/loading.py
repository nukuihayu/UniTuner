"""Load exact native MOSS-Image keys with explicit topology and complete tensor preflight."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from unituner.artifacts.formats.transformer import dequantize_transformer, strip_transformer_prefix
from unituner.artifacts.tensors import save_tensors
from unituner.models.components.state_loading import load_model_state
from unituner.models.components.transformer_artifacts import (
    resolve_transformer_config,
    transformer_tensors,
)
from unituner.models.moss_image.config.model import MossImageConfig
from unituner.models.moss_image.conversion import convert_transformer_tensors
from unituner.models.moss_image.transformer import MossImageTransformer

_CPU = torch.device("cpu")
_CONFIG_KEY = "unituner.moss_image.config"


def save_moss_image(
    model: MossImageTransformer,
    path: Path,
    *,
    metadata: Mapping[str, str] | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    """Atomically export native keys and typed topology, staging one tensor at a time.

    Source parameter identities, gradients, mode and RNG remain unchanged. No
    quantization, QKV fusion or external prefix guessing occurs at this boundary.
    Optional dtype casts only published floating tensors; live weights stay intact.
    """
    save_tensors(
        model.state_dict(),
        path,
        dtype=dtype,
        metadata={**(metadata or {}), _CONFIG_KEY: model.config.model_dump_json()},
    )


def load_moss_image(
    path: Path,
    *,
    config: MossImageConfig | None = None,
    variant: str | None = None,
    version: str | None = None,
    device: torch.device = _CPU,
    dtype: torch.dtype = torch.float32,
    source_format: Literal["auto", "native", "external"] = "auto",
    prefix: Literal["", "model.diffusion_model.", "diffusion_model."] = "",
    fused_qkv: bool = False,
) -> MossImageTransformer:
    """Load native, Diffusers or ComfyUI weights with validated topology, keys and shapes."""
    if source_format not in ("auto", "native", "external"):
        raise ValueError("MOSS source_format must be native or external")
    if source_format == "native" and (prefix or fused_qkv):
        raise ValueError("MOSS prefix/QKV conversion requires external source_format")
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("MOSS native loading supports FP32, FP16 or BF16 weights")
    config = resolve_transformer_config(
        path,
        MossImageConfig,
        metadata_key=_CONFIG_KEY,
        config=config,
        variant=variant,
        version=version,
        mismatch_message="Explicit MOSS topology differs from artifact metadata",
    )
    with torch.device("meta"):
        model = MossImageTransformer(config)
    tensors = transformer_tensors(path)
    if source_format == "auto":
        if prefix:
            if any(not key.startswith(prefix) for key in tensors):
                raise ValueError("MOSS artifact keys lack selected prefix")
            tensors = {key[len(prefix) :]: value for key, value in tensors.items()}
            prefix = ""
        else:
            tensors = strip_transformer_prefix(tensors)
        fused_qkv = fused_qkv or any(key.startswith("layers.") and ".proj.qkv." in key for key in tensors)
        source_format = "external" if fused_qkv or "rope.bands" not in tensors else "native"
    if source_format == "external":
        if fused_qkv:
            tensors = dequantize_transformer(tensors)
        tensors = convert_transformer_tensors(tensors, config, prefix=prefix, fused_qkv=fused_qkv)
    load_model_state(model, tensors, device=device, dtype=dtype, label="MOSS")
    return model
