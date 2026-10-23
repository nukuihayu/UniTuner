"""Strictly import/export native or explicit Diffusers FLUX.2 VAE artifacts."""

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
from unitunner.models.flux2.config.model import Flux2Config
from unitunner.models.flux2.config.vae import Flux2VAEConfig
from unitunner.models.flux2.transformer import Flux2Transformer
from unitunner.models.flux2.vae import Flux2VAE

_CONFIG_KEY = "unitunner.flux2.vae_config"
_CPU = torch.device("cpu")


def save_flux2_vae(vae: Flux2VAE, path: Path) -> None:
    """Atomically export native weights/statistics and reload topology, without mode mutation."""
    save_tensors(vae.state_dict(), path, metadata={_CONFIG_KEY: vae.config.model_dump_json()})


def load_flux2_vae(
    path: Path,
    *,
    config: Flux2VAEConfig | None = None,
    device: torch.device = _CPU,
    source_format: Literal["auto", "native", "diffusers"] = "auto",
    prefix: Literal["", "vae."] = "",
) -> Flux2VAE:
    """Load a complete owned FP32 VAE from single/sharded safetensors.

    Use published topology without metadata or explicit configuration. Auto format
    detects Diffusers keys, reverses decoder blocks and expands 2D attention weights to 1x1
    convolutions. Only the non-numerical BN update counter may be absent in that
    format; initialize it to int64 zero. Reject missing/extra/colliding keys,
    invalid shapes, nonfinite/overflowing weights and negative variance/counters.
    CPU FP32 tensors may retain mmap backing. Source files remain unchanged.
    """
    if source_format not in ("auto", "native", "diffusers") or prefix not in ("", "vae."):
        raise ValueError("FLUX.2 vae requires native/diffusers format and an empty/vae. prefix")
    if source_format == "native" and prefix:
        raise ValueError("FLUX.2 vae prefix conversion requires diffusers format")
    with TensorReader(path) as reader:
        serialized = reader.metadata().get(_CONFIG_KEY)
    embedded = None if serialized is None else Flux2VAEConfig.model_validate_json(serialized)
    if config is None:
        config = Flux2VAEConfig() if embedded is None else embedded
    elif embedded is not None and config != embedded:
        raise ValueError("Explicit FLUX.2 vae topology differs from artifact metadata")
    with torch.device("meta"):
        vae = Flux2VAE(config)
    expected = vae.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    source = dict(iter_tensors(path))
    if source_format == "auto":
        if not prefix and source and all(key.startswith("vae.") for key in source):
            prefix = "vae."
        source_format = "diffusers" if prefix + "encoder.down_blocks.0.resnets.0.norm1.weight" in source else "native"
    for original, tensor in source.items():
        if not original.startswith(prefix):
            raise ValueError(f"FLUX.2 vae key lacks selected prefix: {original}")
        key = original[len(prefix) :]
        if source_format == "diffusers":
            key = native_autoencoder_key(key, len(config.channel_multipliers))
            if ".attn_1." in key and tensor.ndim == 2:
                tensor = tensor[:, :, None, None]
        if key in tensors:
            raise ValueError(f"Duplicate converted FLUX.2 vae tensor: {key}")
        tensors[key] = tensor
    if source_format == "diffusers" and "bn.num_batches_tracked" not in tensors:
        tensors["bn.num_batches_tracked"] = torch.tensor(0, dtype=torch.int64, device="cpu")
    if tensors.keys() != expected.keys():
        raise ValueError(
            f"FLUX.2 vae keys differ: missing={sorted(expected.keys() - tensors.keys())}, extra={sorted(tensors.keys() - expected.keys())}"
        )
    for key, value in tensors.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"FLUX.2 vae tensor shape mismatch: {key}")
        if key == "bn.num_batches_tracked":
            if value.dtype != torch.int64 or value < 0:
                raise ValueError("FLUX.2 vae BN update counter must be a nonnegative int64 scalar")
        elif (
            not value.is_floating_point() or not torch.isfinite(value).all() or not torch.isfinite(value.float()).all()
        ):
            raise ValueError(f"FLUX.2 vae requires finite FP32-convertible weights: {key}")
        if key == "bn.running_var" and (value < 0).any():
            raise ValueError("FLUX.2 vae running variance must be nonnegative")
    vae.load_state_dict(
        {key: value.to(device=device, dtype=expected[key].dtype) for key, value in tensors.items()},
        strict=True,
        assign=True,
    )
    return vae


_TRANSFORMER_CONFIG_KEY = "unitunner.flux2.config"


def save_flux2(
    model: Flux2Transformer, path: Path, *, metadata: Mapping[str, str] | None = None, dtype: torch.dtype | None = None
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
        metadata={**(metadata or {}), _TRANSFORMER_CONFIG_KEY: model.config.model_dump_json()},
    )


def load_flux2(
    path: Path,
    *,
    config: Flux2Config | None = None,
    variant: str | None = None,
    version: str | None = None,
    device: torch.device = _CPU,
    dtype: torch.dtype = torch.float32,
) -> Flux2Transformer:
    """Load published Diffusers/ComfyUI or native single/sharded transformer weights.

    Use embedded topology, an explicit config, a local sidecar or published
    defaults. Validate every key and shape before assigning owned parameters.
    """
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("FLUX.2 native loading supports FP32, FP16 or BF16")
    config = resolve_transformer_config(
        path,
        Flux2Config,
        metadata_key=_TRANSFORMER_CONFIG_KEY,
        config=config,
        variant=variant,
        version=version,
        mismatch_message="Explicit FLUX.2 topology differs from artifact metadata",
    )
    with torch.device("meta"):
        model = Flux2Transformer(config)
    expected, tensors = model.state_dict(), transformer_tensors(path)
    tensors = convert_transformer("flux2", tensors, expected)
    load_model_state(model, tensors, device=device, dtype=dtype, label="FLUX.2")
    return model
