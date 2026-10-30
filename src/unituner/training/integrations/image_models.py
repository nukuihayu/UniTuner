"""Bind loaded native image models to their training contracts."""

from unituner.models.flux.transformer import FluxTransformer
from unituner.models.flux2.transformer import Flux2Transformer
from unituner.models.flux3.transformer import Flux3Transformer
from unituner.models.image import ImageModel
from unituner.models.krea2.transformer import Krea2Transformer
from unituner.models.moss_image.transformer import MossImageTransformer
from unituner.models.qwen_image.transformer import QwenImageTransformer
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unituner.models.z_image.transformer import ZImageTransformer
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
    if isinstance(model, FluxTransformer):
        return FluxBinding(model)
    if isinstance(model, Flux2Transformer):
        return Flux2Binding(model)
    if isinstance(model, Flux3Transformer):
        return Flux3Binding(model)
    if isinstance(model, Krea2Transformer):
        return Krea2Binding(model)
    if isinstance(model, MossImageTransformer):
        return MossImageBinding(model)
    if isinstance(model, QwenImageTransformer):
        return QwenImageBinding(model)
    if isinstance(model, QwenImage21Transformer):
        return QwenImage21Binding(model)
    if isinstance(model, ZImageTransformer):
        return ZImageBinding(model)
    raise TypeError(f"No image training binding for {type(model).__name__}")
