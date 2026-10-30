"""Publish native image previews at completed-update boundaries through the optional media provider."""

from collections.abc import Callable
from importlib import import_module
from typing import Protocol, cast

from torch import nn

from unitunner.config.training import TrainingConfig
from unitunner.data.datasets.restoration import RestorationDataset
from unitunner.models.flux.transformer import FluxTransformer
from unitunner.models.flux2.transformer import Flux2Transformer
from unitunner.models.flux3.transformer import Flux3Transformer
from unitunner.models.image import ImageModel
from unitunner.models.krea2.transformer import Krea2Transformer
from unitunner.models.moss_image.transformer import MossImageTransformer
from unitunner.models.qwen_image.transformer import QwenImageTransformer
from unitunner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unitunner.models.z_image.transformer import ZImageTransformer
from unitunner.precision import compute_context, sampling_precision
from unitunner.training.methods.restoration import RestorationMethod
from unitunner.training.previews.context import preview_context
from unitunner.training.previews.restoration import RestorationPreviews


class PreviewImages:
    """Borrow active training resources and own reusable CPU captions for flow previews.

    Construct without IO. All lazy asset loading and inference occur inside an
    exception-safe preview context. A flow VAE moves to the active device only
    for decode and returns to CPU even on failure. Training modes, Torch/Python/
    NumPy RNG and parameter identities survive previews; no optimizer is touched.
    """

    def __init__(self, config: TrainingConfig, dataset: RestorationDataset | None) -> None:
        if config.preview is None:
            raise ValueError("Image previews require preview configuration")
        self.config = config
        self.dataset = dataset
        self.restoration_preview: RestorationPreviews | None = None
        self.flow_preview: ImagePreview | None = None
        self.flow_model: nn.Module | None = None

    def __call__(
        self,
        update: int,
        model: ImageModel,
        branches: nn.Module,
        method: RestorationMethod | None,
    ) -> None:
        """Publish deterministic PNG previews without advancing the training cursor."""
        if type(update) is not int or update < 0:
            raise ValueError("Preview update must be a nonnegative integer")
        config, preview = (self.config, self.config.preview)
        if preview is None:
            raise ValueError("Image previews require preview configuration")
        device = next(model.parameters()).device
        additional = [branches]
        if method is not None:
            additional.append(method.vae)
            if method.encoder is not None:
                additional.append(method.encoder)
        with preview_context(
            model, additional_models=additional, cuda_devices=(device.index or 0,) if device.type == "cuda" else ()
        ):
            precision = "fp32" if config.execution.precision == "auto" else config.execution.precision
            with sampling_precision(config.execution.precision), compute_context(device, precision):
                if method is not None:
                    if self.dataset is None:
                        raise ValueError("Restoration previews require indexed dataset items")
                    if self.restoration_preview is None:
                        self.restoration_preview = RestorationPreviews(config, self.dataset)
                    self.restoration_preview(update, method, device)
                    return
                if self.flow_preview is None or self.flow_model is not model:
                    self.flow_preview = build_flow_preview(config, model)
                    self.flow_model = model
                self.flow_preview(update)


class ImagePreview(Protocol):
    """Publish a completed update while borrowing a live model; construction performs no IO."""

    def __call__(self, update: int) -> None:
        """Publish samples inside the caller-owned preview context."""
        ...


def build_flow_preview(
    config: TrainingConfig,
    model: ImageModel,
) -> ImagePreview:
    """Select the native image publisher explicitly without a fallback model family."""
    providers = (
        (FluxTransformer, "flux"),
        (Flux2Transformer, "flux2"),
        (Flux3Transformer, "flux3"),
        (MossImageTransformer, "moss_image"),
        (ZImageTransformer, "z_image"),
        (QwenImageTransformer, "qwen_image"),
        (QwenImage21Transformer, "qwen_image_2_1"),
        (Krea2Transformer, "krea2"),
    )
    for model_type, family in providers:
        if isinstance(model, model_type):
            factory = cast(
                Callable[[TrainingConfig, ImageModel], ImagePreview],
                import_module(f"unitunner.training.integrations.{family}.previews").ImagePreviews,
            )
            return factory(config, model)
    raise TypeError("No image preview publisher for this model type")
