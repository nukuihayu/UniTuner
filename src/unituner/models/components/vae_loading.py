"""Read released VAE checkpoints and normalize documented component prefixes."""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from unituner.artifacts.checkpoint_reading import checkpoint_tensors
from unituner.artifacts.formats.autoencoder import native_autoencoder_key
from unituner.artifacts.tensors import iter_tensors


def vae_tensors(path: Path, *, prefixes: tuple[str, ...] = ()) -> Iterator[tuple[str, Tensor]]:
    """Read safetensors or Wan PyTorch weights, rejecting prefix collisions."""
    seen: set[str] = set()
    for name, tensor in checkpoint_tensors(path, label="VAE"):
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise ValueError("VAE checkpoint must contain named tensors")
        key = name
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key in seen:
            raise ValueError(f"VAE keys collide after prefix removal: {key}")
        seen.add(key)
        yield key, tensor


def flux_vae_tensors(
    path: Path, *, stages: int, source_format: Literal["auto", "native", "diffusers"], prefix: str, label: str
) -> tuple[dict[str, Tensor], Literal["native", "diffusers"]]:
    """Read native or Diffusers FLUX VAE keys, preserving tensor views and detecting collisions."""
    source = dict(iter_tensors(path))
    if source_format == "auto":
        if not prefix and source and all(key.startswith("vae.") for key in source):
            prefix = "vae."
        source_format = "diffusers" if prefix + "encoder.down_blocks.0.resnets.0.norm1.weight" in source else "native"
    tensors = {}
    for original, value in source.items():
        if not original.startswith(prefix):
            raise ValueError(f"{label} key lacks selected prefix: {original}")
        key = original[len(prefix) :]
        if source_format == "diffusers":
            key = native_autoencoder_key(key, stages)
            if ".attn_1." in key and value.ndim == 2:
                value = value[:, :, None, None]
        if key in tensors:
            raise ValueError(f"Duplicate converted {label} tensor: {key}")
        tensors[key] = value
    return tensors, source_format


def load_vae_state(
    vae: nn.Module,
    tensors: Mapping[str, Tensor],
    *,
    device: torch.device,
    label: str,
    nonnegative: tuple[str, ...] = (),
) -> None:
    """Validate the complete VAE state before assigning FP32 weights and native integer buffers."""
    expected = vae.state_dict()
    if expected.keys() != tensors.keys():
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected.keys() - tensors.keys())}, "
            f"extra={sorted(tensors.keys() - expected.keys())}"
        )
    for name, value in tensors.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"{label} invalid tensor shape/type; values invalid: {name}")
        if expected[name].is_floating_point():
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{label} invalid tensor shape/type; values invalid: {name}")
            if value.dtype != torch.float32 and not torch.isfinite(value.float()).all():
                raise ValueError(f"{label} tensor exceeds FP32 range; values invalid: {name}")
        elif value.dtype != expected[name].dtype:
            raise ValueError(f"{label} invalid tensor type: {name}")
        if name in nonnegative and (value < 0).any():
            raise ValueError(f"{label} tensor must be nonnegative: {name}")
    vae.load_state_dict(
        {
            name: value.to(
                device=device, dtype=torch.float32 if expected[name].is_floating_point() else expected[name].dtype
            )
            for name, value in tensors.items()
        },
        strict=True,
        assign=True,
    )
