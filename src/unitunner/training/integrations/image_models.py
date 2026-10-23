"""Bind native image model loading to training contracts."""

from pathlib import Path

import torch

from unitunner.config.model import ModelConfig
from unitunner.models.catalog import ModelFamily
from unitunner.models.flux.loading import load_flux
from unitunner.models.flux2.loading import load_flux2
from unitunner.models.flux3.loading import load_flux3
from unitunner.models.image import ImageModel
from unitunner.models.krea2.loading import load_krea2
from unitunner.models.moss_image.loading import load_moss_image
from unitunner.models.qwen_image.loading import load_qwen_image
from unitunner.models.qwen_image_2_1.loading import load_qwen_image_21
from unitunner.models.z_image.loading import load_z_image
from unitunner.training.integrations.flux.binding import FluxBinding
from unitunner.training.integrations.flux2.binding import Flux2Binding
from unitunner.training.integrations.flux3.binding import Flux3Binding
from unitunner.training.integrations.krea2.binding import Krea2Binding
from unitunner.training.integrations.moss_image.binding import MossImageBinding
from unitunner.training.integrations.qwen_image.binding import QwenImageBinding
from unitunner.training.integrations.qwen_image_2_1.binding import QwenImage21Binding
from unitunner.training.integrations.z_image.binding import ZImageBinding

type ImageBinding = (
    FluxBinding
    | Flux2Binding
    | Flux3Binding
    | Krea2Binding
    | MossImageBinding
    | QwenImageBinding
    | QwenImage21Binding
    | ZImageBinding
)


def load_image_model(
    config: ModelConfig, *, device: torch.device, dtype: torch.dtype
) -> tuple[ImageModel, ImageBinding]:
    """Construct the declared family and validate model-specific artifact versions."""
    if config.family == ModelFamily.FLUX:
        model = load_flux(Path(config.weights), variant=config.variant, device=device, dtype=dtype)
        binding = FluxBinding(model)
    elif config.family == ModelFamily.FLUX2:
        model = load_flux2(
            Path(config.weights),
            variant=config.variant,
            version=config.version,
            device=device,
            dtype=dtype,
        )
        binding = Flux2Binding(model)
        if model.config.version != config.version:
            raise ValueError("Training FLUX.2 version differs from artifact topology")
    elif config.family == ModelFamily.FLUX3:
        model = load_flux3(Path(config.weights), variant=config.variant, device=device, dtype=dtype)
        binding = Flux3Binding(model)
    elif config.family == ModelFamily.KREA2:
        model = load_krea2(Path(config.weights), variant=config.variant, device=device, dtype=dtype)
        binding = Krea2Binding(model)
    elif config.family == ModelFamily.MOSS_IMAGE:
        model = load_moss_image(Path(config.weights), variant=config.variant, device=device, dtype=dtype)
        binding = MossImageBinding(model)
    elif config.family == ModelFamily.QWEN_IMAGE:
        model = load_qwen_image(Path(config.weights), variant=config.variant, device=device, dtype=dtype)
        binding = QwenImageBinding(model)
    elif config.family == ModelFamily.QWEN_IMAGE_2_1:
        model = load_qwen_image_21(Path(config.weights), device=device, dtype=dtype)
        binding = QwenImage21Binding(model)
    elif config.family == ModelFamily.Z_IMAGE:
        model = load_z_image(Path(config.weights), variant=config.variant, device=device, dtype=dtype)
        binding = ZImageBinding(model)
    else:
        raise ValueError(f"No image training integration for {config.family}")
    return model, binding
