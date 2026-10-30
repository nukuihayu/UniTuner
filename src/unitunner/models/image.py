"""Select native image denoisers without training or inference policy."""

from pathlib import Path

import torch

from unitunner.config.model import ModelConfig
from unitunner.models.catalog import ModelFamily
from unitunner.models.flux.loading import load_flux
from unitunner.models.flux.transformer import FluxTransformer
from unitunner.models.flux2.loading import load_flux2
from unitunner.models.flux2.transformer import Flux2Transformer
from unitunner.models.flux3.loading import load_flux3
from unitunner.models.flux3.transformer import Flux3Transformer
from unitunner.models.krea2.loading import load_krea2
from unitunner.models.krea2.transformer import Krea2Transformer
from unitunner.models.moss_image.loading import load_moss_image
from unitunner.models.moss_image.transformer import MossImageTransformer
from unitunner.models.qwen_image.loading import load_qwen_image
from unitunner.models.qwen_image.transformer import QwenImageTransformer
from unitunner.models.qwen_image_2_1.loading import load_qwen_image_21
from unitunner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unitunner.models.z_image.loading import load_z_image
from unitunner.models.z_image.transformer import ZImageTransformer

type ImageModel = (
    FluxTransformer
    | Flux2Transformer
    | Flux3Transformer
    | Krea2Transformer
    | MossImageTransformer
    | QwenImageTransformer
    | QwenImage21Transformer
    | ZImageTransformer
)


def load_image_model(config: ModelConfig, *, device: torch.device, dtype: torch.dtype) -> ImageModel:
    """Load the declared family, retaining native variant and version validation."""
    path = Path(config.weights)
    match config.family:
        case ModelFamily.FLUX:
            return load_flux(path, variant=config.variant, device=device, dtype=dtype)
        case ModelFamily.FLUX2:
            model = load_flux2(path, variant=config.variant, version=config.version, device=device, dtype=dtype)
            if model.config.version != config.version:
                raise ValueError("FLUX.2 version differs from artifact topology")
            return model
        case ModelFamily.FLUX3:
            return load_flux3(path, variant=config.variant, device=device, dtype=dtype)
        case ModelFamily.KREA2:
            return load_krea2(path, variant=config.variant, device=device, dtype=dtype)
        case ModelFamily.MOSS_IMAGE:
            return load_moss_image(path, variant=config.variant, device=device, dtype=dtype)
        case ModelFamily.QWEN_IMAGE:
            return load_qwen_image(path, variant=config.variant, device=device, dtype=dtype)
        case ModelFamily.QWEN_IMAGE_2_1:
            return load_qwen_image_21(path, device=device, dtype=dtype)
        case ModelFamily.Z_IMAGE:
            return load_z_image(path, variant=config.variant, device=device, dtype=dtype)
        case _:
            raise ValueError(f"No image model for {config.family}")
