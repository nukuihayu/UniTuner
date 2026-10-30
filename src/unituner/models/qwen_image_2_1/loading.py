"""Strictly load and atomically publish native Qwen Image 2.1 model artifacts."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from unituner.artifacts.formats.qwen_image import convert_vae_tensors, image21_vae_key_map, remove_prefix
from unituner.artifacts.formats.transformer import convert_transformer
from unituner.artifacts.tensors import TensorReader, save_tensors
from unituner.models.components.state_loading import load_model_state
from unituner.models.components.transformer_artifacts import (
    resolve_transformer_config,
    transformer_tensors,
)
from unituner.models.components.vae_loading import load_vae_state, vae_tensors
from unituner.models.qwen_image_2_1.config.model import QwenImage21Config
from unituner.models.qwen_image_2_1.config.vae import QwenImage21VAEConfig
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unituner.models.qwen_image_2_1.vae import QwenImage21VAE

_CONFIG_KEY = "unituner.qwen_image_2_1.config"
_CPU = torch.device("cpu")


def save_qwen_image_21(
    model: QwenImage21Transformer,
    path: Path,
    *,
    metadata: Mapping[str, str] | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    """Atomically export native weights and topology, casting only published floating tensors."""
    save_tensors(
        model.state_dict(),
        path,
        dtype=dtype,
        metadata={**({} if metadata is None else metadata), _CONFIG_KEY: model.config.model_dump_json()},
    )


def load_qwen_image_21(
    path: Path,
    *,
    config: QwenImage21Config | None = None,
    variant: str | None = None,
    version: str | None = None,
    device: torch.device = _CPU,
    dtype: torch.dtype = torch.float32,
) -> QwenImage21Transformer:
    """Load native, Diffusers or ComfyUI weights with validated topology, keys and shapes."""
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Qwen Image 2.1 weights require FP32/FP16/BF16 loading")
    config = resolve_transformer_config(
        path,
        QwenImage21Config,
        metadata_key=_CONFIG_KEY,
        config=config,
        variant=variant,
        version=version,
        mismatch_message="Qwen Image 2.1 explicit topology differs from artifact metadata",
    )
    with torch.device("meta"):
        model = QwenImage21Transformer(config)
    expected, tensors = model.state_dict(), transformer_tensors(path)
    tensors = convert_transformer("qwen_image_2_1", tensors, expected)
    load_model_state(model, tensors, device=device, dtype=dtype, label="Qwen Image2.1")
    return model


def save_qwen_image_21_vae(vae: QwenImage21VAE, path: Path) -> None:
    """Atomically publish native image VAE tensors and full RGB/RGBA topology."""
    save_tensors(vae.state_dict(), path, metadata={"unituner.qwen_image_2_1.vae_config": vae.config.model_dump_json()})


def load_qwen_image_21_vae(
    path: Path,
    *,
    config: QwenImage21VAEConfig | None = None,
    device: torch.device = _CPU,
    source_format: Literal["auto", "native", "diffusers", "comfyui"] = "auto",
    prefix: Literal["", "vae."] = "",
) -> QwenImage21VAE:
    """Load native or released Diffusers/ComfyUI VAE weights into image-only FP32 layers.

    Explicit/embedded topology must agree. Meta construction and exact assignment
    preserve caller RNG. External singleton temporal kernels become spatial views;
    all weights, including inactive temporal convolutions, remain strictly validated.
    """
    if source_format not in ("auto", "native", "diffusers", "comfyui") or prefix not in ("", "vae."):
        raise ValueError("Qwen Image2.1 vae requires native/diffusers/comfyui format and empty/vae. prefix")
    if source_format == "native" and prefix:
        raise ValueError("Qwen Image2.1 native vae does not use a prefix")
    with TensorReader(path) as reader:
        serialized = reader.metadata().get("unituner.qwen_image_2_1.vae_config")
    embedded = None if serialized is None else QwenImage21VAEConfig.model_validate_json(serialized)
    if config is None:
        config = QwenImage21VAEConfig() if embedded is None else embedded
    elif embedded is not None and config != embedded:
        raise ValueError("Qwen Image2.1 vae topology differs from artifact metadata")
    with torch.device("meta"):
        vae = QwenImage21VAE(config)
    expected, tensors = vae.state_dict(), dict(vae_tensors(path))
    if source_format == "auto" and not prefix and tensors and all(key.startswith("vae.") for key in tensors):
        prefix = "vae."
    tensors = remove_prefix(tensors, prefix)
    if source_format == "comfyui" or (source_format == "auto" and "conv1.bias" in tensors):
        tensors = convert_vae_tensors(
            tensors,
            image21_vae_key_map(
                tuple(expected),
                num_blocks=len(config.channel_multipliers),
                layers_per_block=config.layers_per_block,
                is_residual=config.is_residual,
            ),
            to_native=True,
        )
    if source_format != "native":
        tensors = {
            name: value.squeeze(2)
            if name in expected and value.ndim == 5 and expected[name].ndim == 4 and value.shape[2] == 1
            else value
            for name, value in tensors.items()
        }
    load_vae_state(vae, tensors, device=device, label="Qwen Image2.1 vae")
    return vae
