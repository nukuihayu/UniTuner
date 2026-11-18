"""Load exact native Z-Image artifacts without training or external providers."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from unituner.artifacts.formats.transformer import convert_transformer
from unituner.artifacts.formats.z_image import convert_vae_tensors, vae_key_map
from unituner.artifacts.tensors import TensorReader, iter_tensors, save_tensors
from unituner.models.components.state_loading import load_model_state
from unituner.models.components.transformer_artifacts import (
    resolve_transformer_config,
    transformer_tensors,
)
from unituner.models.components.vae_loading import load_vae_state
from unituner.models.z_image.config.model import ZImageConfig
from unituner.models.z_image.config.vae import ZImageVAEConfig
from unituner.models.z_image.transformer import ZImageTransformer
from unituner.models.z_image.vae import ZImageVAE

_CPU = torch.device("cpu")
_CONFIG_KEY = "unituner.z_image.config"


def save_z_image(
    model: ZImageTransformer, path: Path, *, metadata: Mapping[str, str] | None = None, dtype: torch.dtype | None = None
) -> None:
    """Atomically export exact native keys and reload topology in safetensors metadata.

    Stage at most one detached tensor on CPU; the caller owns the live model.
    Export does not change parameter identities, RNG or training mode.
    Optional dtype casts only published floating tensors; live weights stay intact.
    """
    if metadata is not None and _CONFIG_KEY in metadata:
        raise ValueError("Additional model metadata must not override native topology")
    save_tensors(
        model.state_dict(),
        path,
        dtype=dtype,
        metadata={**(metadata or {}), _CONFIG_KEY: model.config.model_dump_json()},
    )


def load_z_image(
    path: Path,
    *,
    config: ZImageConfig | None = None,
    variant: str | None = None,
    version: str | None = None,
    device: torch.device = _CPU,
    dtype: torch.dtype | None = None,
) -> ZImageTransformer:
    """Load native, Diffusers or ComfyUI weights with validated topology, keys and shapes."""
    if dtype is not None and not dtype.is_floating_point:
        raise ValueError("Z-Image load dtype must be floating")
    config = resolve_transformer_config(
        path,
        ZImageConfig,
        metadata_key=_CONFIG_KEY,
        config=config,
        variant=variant,
        version=version,
        mismatch_message="Explicit topology differs from Z-Image artifact metadata",
    )
    with torch.device("meta"):
        model = ZImageTransformer(config)
    expected = model.state_dict()
    tensors = transformer_tensors(path)
    tensors = convert_transformer("z_image", tensors, expected)
    load_model_state(model, tensors, device=device, dtype=dtype, label="Z-Image", check_finite=False)
    return model


def save_z_image_vae(vae: ZImageVAE, path: Path) -> None:
    """Atomically export native VAE keys with exact topology and normalization."""
    save_tensors(vae.state_dict(), path, metadata={"unituner.z_image.vae_config": vae.config.model_dump_json()})


def load_z_image_vae(
    path: Path,
    *,
    config: ZImageVAEConfig | None = None,
    device: torch.device = _CPU,
    source_format: Literal["auto", "native", "comfyui"] = "auto",
) -> ZImageVAE:
    """Load exact native/Diffusers-layout VAE weights as owned FP32 parameters.

    External files use the published Z-Image topology by default. Auto format
    recognizes a uniform vae. prefix and Musubi's ComfyUI sentinel. All keys and shapes must match bijectively.
    CPU values may retain immutable mmap storage; input artifacts are not changed.
    """
    if source_format not in {"auto", "native", "comfyui"}:
        raise ValueError("Z-Image vae source_format must be auto, native or comfyui")
    with TensorReader(path) as reader:
        serialized = reader.metadata().get("unituner.z_image.vae_config")
    embedded = None if serialized is None else ZImageVAEConfig.model_validate_json(serialized)
    if config is None:
        config = ZImageVAEConfig() if embedded is None else embedded
    elif embedded is not None and embedded != config:
        raise ValueError("Explicit topology differs from Z-Image vae artifact metadata")
    with torch.device("meta"):
        vae = ZImageVAE(config)
    expected = vae.state_dict()
    tensors = dict(iter_tensors(path))
    if source_format == "auto":
        if tensors and all(key.startswith("vae.") for key in tensors):
            tensors = {key.removeprefix("vae."): value for key, value in tensors.items()}
        source_format = "comfyui" if "encoder.down.0.block.0.conv1.weight" in tensors else "native"
    if source_format == "comfyui":
        tensors = convert_vae_tensors(
            tensors, vae_key_map(tuple(expected), num_blocks=len(config.block_channels)), to_native=True
        )
    load_vae_state(vae, tensors, device=device, label="Z-Image vae")
    return vae
