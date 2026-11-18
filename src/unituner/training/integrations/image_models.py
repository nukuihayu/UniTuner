"""Bind loaded native image models to their training contracts."""

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from unituner.models.image import ImageModel, model_spec
from unituner.models.providers import load_provider

if TYPE_CHECKING:
    from unituner.training.integrations.flux.binding import FluxBinding
    from unituner.training.integrations.flux2.binding import Flux2Binding
    from unituner.training.integrations.flux3.binding import Flux3Binding
    from unituner.training.integrations.krea2.binding import Krea2Binding
    from unituner.training.integrations.moss_image.binding import MossImageBinding
    from unituner.training.integrations.qwen_image.binding import QwenImageBinding
    from unituner.training.integrations.qwen_image_2_1.binding import QwenImage21Binding
    from unituner.training.integrations.z_image.binding import ZImageBinding

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


def bind_image_model(model: ImageModel) -> ImageBinding:
    """Borrow a native model without loading or changing its weights."""
    factory = cast(Callable[[ImageModel], ImageBinding], load_provider(model_spec(model).binding))
    return factory(model)
