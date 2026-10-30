"""Load native FLUX.1 safetensors with explicit topology and complete validation."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from unituner.artifacts.formats.transformer import convert_transformer
from unituner.artifacts.tensors import TensorReader, save_tensors
from unituner.models.components.state_loading import load_model_state
from unituner.models.components.transformer_artifacts import (
    resolve_transformer_config,
    transformer_tensors,
)
from unituner.models.components.vae_loading import flux_vae_tensors, load_vae_state
from unituner.models.flux.config.model import FluxConfig
from unituner.models.flux.config.vae import FluxVAEConfig
from unituner.models.flux.transformer import FluxTransformer
from unituner.models.flux.vae import FluxVAE

_CPU = torch.device("cpu")
_CONFIG_KEY = "unituner.flux.config"


def save_flux(
    model: FluxTransformer, path: Path, *, metadata: Mapping[str, str] | None = None, dtype: torch.dtype | None = None
) -> None:
    """Atomically save native keys/topology without modifying live parameters or RNG.

    Values are staged one tensor at a time. No external prefix or fused projection
    conversion is inferred; the native published QKV layout is preserved.
    Optional dtype casts only published floating tensors; live weights stay intact.
    """
    save_tensors(
        model.state_dict(),
        path,
        dtype=dtype,
        metadata={**(metadata or {}), _CONFIG_KEY: model.config.model_dump_json()},
    )


def load_flux(
    path: Path,
    *,
    config: FluxConfig | None = None,
    variant: str | None = None,
    version: str | None = None,
    device: torch.device = _CPU,
    dtype: torch.dtype = torch.float32,
) -> FluxTransformer:
    """Load native, Diffusers or ComfyUI weights with validated topology, keys and shapes."""
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("FLUX native loading supports FP32, FP16 or BF16")
    config = resolve_transformer_config(
        path,
        FluxConfig,
        metadata_key=_CONFIG_KEY,
        config=config,
        variant=variant,
        version=version,
        mismatch_message="Explicit FLUX topology differs from artifact metadata",
    )
    with torch.device("meta"):
        model = FluxTransformer(config)
    expected, tensors = model.state_dict(), transformer_tensors(path)
    tensors = convert_transformer("flux", tensors, expected)
    load_model_state(model, tensors, device=device, dtype=dtype, label="FLUX")
    return model


_VAE_CONFIG_KEY = "unituner.flux.vae_config"


def save_flux_vae(vae: FluxVAE, path: Path) -> None:
    """Atomically save native VAE keys/topology without changing modes or RNG."""
    save_tensors(vae.state_dict(), path, metadata={_VAE_CONFIG_KEY: vae.config.model_dump_json()})


def load_flux_vae(
    path: Path,
    *,
    config: FluxVAEConfig | None = None,
    device: torch.device = _CPU,
    source_format: Literal["auto", "native", "diffusers"] = "auto",
    prefix: Literal["", "vae."] = "",
) -> FluxVAE:
    """Load an owned complete FP32 native VAE from single/sharded safetensors.

    Use published topology without metadata or explicit configuration. Diffusers conversion
    reverses decoder indices using actual stage count and expands linear spatial
    attention weights into 1x1 convolutions. Reject missing/extra/colliding keys,
    nonfloating/nonfinite values, shape mismatches and FP32 overflow. Auto format
    recognizes a uniform vae. prefix; unknown tensors fail. CPU values may retain mmap backing.
    """
    if source_format not in ("auto", "native", "diffusers") or prefix not in ("", "vae."):
        raise ValueError("FLUX vae requires native/diffusers format and an empty/vae. prefix")
    if source_format == "native" and prefix:
        raise ValueError("FLUX vae prefix conversion requires diffusers format")
    with TensorReader(path) as reader:
        serialized = reader.metadata().get(_VAE_CONFIG_KEY)
    embedded = None if serialized is None else FluxVAEConfig.model_validate_json(serialized)
    if config is None:
        config = FluxVAEConfig() if embedded is None else embedded
    elif embedded is not None and embedded != config:
        raise ValueError("Explicit FLUX vae topology differs from artifact metadata")
    with torch.device("meta"):
        vae = FluxVAE(config)
    tensors, source_format = flux_vae_tensors(
        path,
        stages=len(config.channel_multipliers),
        source_format=source_format,
        prefix=prefix,
        label="FLUX vae",
    )
    load_vae_state(vae, tensors, device=device, label="FLUX vae")
    return vae
