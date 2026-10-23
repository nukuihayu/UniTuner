"""Load native FLUX.1 safetensors with explicit topology and complete validation."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from unitunner.artifacts.formats.autoencoder import native_autoencoder_key
from unitunner.artifacts.formats.transformer import convert_transformer
from unitunner.artifacts.tensors import TensorReader, iter_tensors, save_tensors
from unitunner.models.components.state_loading import load_model_state
from unitunner.models.components.transformer_artifacts import (
    resolve_transformer_config,
    transformer_tensors,
)
from unitunner.models.flux.config.model import FluxConfig
from unitunner.models.flux.config.vae import FluxVAEConfig
from unitunner.models.flux.transformer import FluxTransformer
from unitunner.models.flux.vae import FluxVAE

_CPU = torch.device("cpu")
_CONFIG_KEY = "unitunner.flux.config"


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
    """Load published Diffusers/ComfyUI or native single/sharded transformer weights.

    Use embedded topology, an explicit config, a local sidecar or published
    defaults. Validate every key and shape before assigning owned parameters.
    """
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


_VAE_CONFIG_KEY = "unitunner.flux.vae_config"


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
    expected, tensors = vae.state_dict(), {}
    source = dict(iter_tensors(path))
    if source_format == "auto":
        if not prefix and source and all(key.startswith("vae.") for key in source):
            prefix = "vae."
        source_format = "diffusers" if prefix + "encoder.down_blocks.0.resnets.0.norm1.weight" in source else "native"
    for original, value in source.items():
        if not original.startswith(prefix):
            raise ValueError(f"FLUX vae key lacks selected prefix: {original}")
        key = original[len(prefix) :]
        if source_format == "diffusers":
            key = native_autoencoder_key(key, len(config.channel_multipliers))
            if ".attn_1." in key and value.ndim == 2:
                value = value[:, :, None, None]
        if key in tensors:
            raise ValueError(f"Duplicate converted FLUX vae tensor: {key}")
        tensors[key] = value
    if tensors.keys() != expected.keys():
        raise ValueError(
            f"FLUX vae keys differ: missing={sorted(expected.keys() - tensors.keys())}, extra={sorted(tensors.keys() - expected.keys())}"
        )
    for key, value in tensors.items():
        if (
            value.shape != expected[key].shape
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
            or not torch.isfinite(value.float()).all()
        ):
            raise ValueError(f"FLUX vae tensor shape/type/values invalid: {key}")
    vae.load_state_dict(
        {key: value.to(device=device, dtype=torch.float32) for key, value in tensors.items()}, strict=True, assign=True
    )
    return vae
