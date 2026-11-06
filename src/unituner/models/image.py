"""Select native image denoisers without training or inference policy."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from unituner.config.model import ModelConfig
from unituner.models.catalog import ImageFamilySpec, image_family_spec
from unituner.models.providers import load_provider

if TYPE_CHECKING:
    from unituner.models.bluepixel.transformer import BluePixelTransformer
    from unituner.models.flux.transformer import FluxTransformer
    from unituner.models.flux2.transformer import Flux2Transformer
    from unituner.models.flux3.transformer import Flux3Transformer
    from unituner.models.krea2.transformer import Krea2Transformer
    from unituner.models.moss_image.transformer import MossImageTransformer
    from unituner.models.qwen_image.transformer import QwenImageTransformer
    from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
    from unituner.models.z_image.transformer import ZImageTransformer

type ImageModel = (
    FluxTransformer
    | Flux2Transformer
    | Flux3Transformer
    | Krea2Transformer
    | MossImageTransformer
    | QwenImageTransformer
    | QwenImage21Transformer
    | ZImageTransformer
    | BluePixelTransformer
)


def load_image_model(config: ModelConfig, *, device: torch.device, dtype: torch.dtype) -> ImageModel:
    """Load the declared family, retaining native variant and version validation."""
    spec = image_family_spec(config.family)
    arguments = {}
    if spec.loader_variant:
        arguments["variant"] = spec.selected_variant(config.variant)
    if spec.loader_version:
        arguments["version"] = spec.selected_version(config.version)
    model = load_provider(spec.loader)(Path(config.weights), device=device, dtype=dtype, **arguments)
    validate_model_selection(config, model)
    return model


def model_spec(model: ImageModel) -> ImageFamilySpec:
    """Read exact public identity, including derivatives sharing their parent's topology."""
    return image_family_spec(model.family)


def validate_model_selection(config: ModelConfig, model: ImageModel) -> None:
    """Check public identity and declared native variant/version mappings together."""
    spec = image_family_spec(config.family)
    if model.family != spec.family:
        raise ValueError("Model family differs from the selected provider")
    if spec.variant_in_weights and model.config.variant != spec.selected_variant(config.variant):
        raise ValueError("Model variant differs from the selected native topology")
    if (spec.loader_version or spec.native_version is not None) and model.config.version != spec.selected_version(
        config.version
    ):
        raise ValueError("Model version differs from the selected native topology")


def model_latent_channels(model: ImageModel) -> int:
    """Describe transformer-visible latent channels independently of the VAE implementation."""
    spec = model_spec(model)
    channels = getattr(model.config, spec.latent_channels_field)
    if type(channels) is not int or channels <= 0 or channels % spec.latent_channels_divisor:
        raise ValueError("Model has invalid latent channel geometry")
    return channels // spec.latent_channels_divisor


def image_adapter_targets(model: ImageModel) -> dict[str, torch.nn.Linear]:
    """Borrow model-owned projections through the declared backbone provider."""
    targets = cast(Callable[[ImageModel], dict[str, torch.nn.Linear]], load_provider(model_spec(model).projections))
    return targets(model)


def save_image_model(model: ImageModel, path: Path, *, dtype: torch.dtype, metadata: Mapping[str, str]) -> None:
    """Publish through the provider owning this exact model identity."""
    load_provider(model_spec(model).saver)(model, path, dtype=dtype, metadata=metadata)
