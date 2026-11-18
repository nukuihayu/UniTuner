"""Strictly load and publish native Qwen Image transformer topology and tensors."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from unituner.artifacts.formats.qwen_image import (
    convert_transformer_tensors,
    convert_vae_tensors,
    remove_prefix,
    vae_key_map,
)
from unituner.artifacts.formats.transformer import strip_transformer_prefix
from unituner.artifacts.tensors import TensorReader, save_tensors
from unituner.models.components.state_loading import load_model_state
from unituner.models.components.transformer_artifacts import (
    resolve_transformer_config,
    transformer_tensors,
)
from unituner.models.components.vae_loading import load_vae_state, vae_tensors
from unituner.models.qwen_image.config.model import QwenImageConfig
from unituner.models.qwen_image.config.vae import QwenImageVAEConfig
from unituner.models.qwen_image.transformer import QwenImageTransformer
from unituner.models.qwen_image.vae import QwenImageVAE

_CONFIG_KEY = "unituner.qwen_image.config"
_CPU = torch.device("cpu")


def save_qwen_image_vae(vae: QwenImageVAE, path: Path) -> None:
    """Atomically export native VAE tensors and explicit RGB/RGBA topology."""
    save_tensors(vae.state_dict(), path, metadata={"unituner.qwen_image.vae_config": vae.config.model_dump_json()})


def load_qwen_image_vae(
    path: Path,
    *,
    config: QwenImageVAEConfig | None = None,
    device: torch.device = _CPU,
    source_format: Literal["auto", "native", "comfyui"] = "auto",
    prefix: Literal["", "vae."] = "",
    input_channels: Literal[3, 4] | None = None,
) -> QwenImageVAE:
    """Load owned FP32 VAE parameters with exact topology, keys and finite values.

    Construction on meta consumes no RNG. External artifacts use the published
    RGB/RGBA topology. Auto format accepts official Qwen/Wan or ComfyUI keys;
    embedded and explicit configurations must agree. CPU tensors may retain mmap backing.
    """
    if (
        source_format not in ("auto", "native", "comfyui")
        or prefix not in ("", "vae.")
        or (source_format == "native" and prefix)
    ):
        raise ValueError("Qwen vae conversion requires explicit comfyui format and empty/vae. prefix")
    serialized = None
    if path.suffix.lower() not in (".pt", ".pth"):
        with TensorReader(path) as reader:
            serialized = reader.metadata().get("unituner.qwen_image.vae_config")
    embedded = None if serialized is None else QwenImageVAEConfig.model_validate_json(serialized)
    if config is None:
        config = (
            QwenImageVAEConfig(input_channels=3 if input_channels is None else input_channels)
            if embedded is None
            else embedded
        )
    elif embedded is not None and config != embedded:
        raise ValueError("Qwen Image vae explicit topology differs from artifact metadata")
    if input_channels is not None and config.input_channels != input_channels:
        raise ValueError("Qwen Image vae input channels differ from the requested variant")
    with torch.device("meta"):
        vae = QwenImageVAE(config)
    tensors = dict(vae_tensors(path))
    if source_format == "auto":
        if not prefix and tensors and all(key.startswith("vae.") for key in tensors):
            prefix = "vae."
        tensors = remove_prefix(tensors, prefix)
        prefix = ""
        source_format = "comfyui" if "conv1.bias" in tensors else "native"
    if source_format == "comfyui":
        tensors = convert_vae_tensors(
            remove_prefix(tensors, prefix),
            vae_key_map(
                tuple(vae.state_dict()),
                num_blocks=len(config.channel_multipliers),
                layers_per_block=config.layers_per_block,
            ),
            to_native=True,
        )
    load_vae_state(vae, tensors, device=device, label="Qwen Image vae")
    return vae


def save_qwen_image(
    model: QwenImageTransformer,
    path: Path,
    *,
    metadata: Mapping[str, str] | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    """Atomically export native weight keys and complete immutable reload topology.

    Optional dtype casts only published floating tensors; live weights stay intact.
    """
    if metadata is not None and _CONFIG_KEY in metadata:
        raise ValueError("Additional model metadata must not replace Qwen topology")
    save_tensors(
        model.state_dict(),
        path,
        dtype=dtype,
        metadata={**(metadata or {}), _CONFIG_KEY: model.config.model_dump_json()},
    )


def load_qwen_image(
    path: Path,
    *,
    config: QwenImageConfig | None = None,
    variant: str | None = None,
    version: str | None = None,
    device: torch.device = _CPU,
    dtype: torch.dtype = torch.float32,
    source_format: Literal["auto", "native", "diffusers", "comfyui"] = "auto",
    prefix: Literal["", "model.diffusion_model.", "diffusion_model."] = "",
) -> QwenImageTransformer:
    """Load native, Diffusers or ComfyUI weights with validated topology, keys and shapes."""
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Qwen Image weights support FP32/FP16/BF16 floating loading")
    if (
        source_format not in ("auto", "native", "diffusers", "comfyui")
        or prefix not in ("", "model.diffusion_model.", "diffusion_model.")
        or (source_format not in ("auto", "comfyui") and prefix)
    ):
        raise ValueError("Qwen transformer prefix conversion requires explicit comfyui format")
    config = resolve_transformer_config(
        path,
        QwenImageConfig,
        metadata_key=_CONFIG_KEY,
        config=config,
        variant=variant,
        version=version,
        mismatch_message="Qwen Image explicit topology differs from artifact metadata",
    )
    with torch.device("meta"):
        model = QwenImageTransformer(config)
    tensors = transformer_tensors(path)
    if source_format == "auto":
        if not prefix:
            tensors = strip_transformer_prefix(tensors)
        source_format = "comfyui"
    if source_format == "comfyui":
        tensors = convert_transformer_tensors(tensors, prefix=prefix, variant=config.variant)
    load_model_state(model, tensors, device=device, dtype=dtype, label="Qwen Image")
    return model
