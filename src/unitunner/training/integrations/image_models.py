"""Bind loaded native image models to their training contracts."""

from unitunner.models.flux.transformer import FluxTransformer
from unitunner.models.flux2.transformer import Flux2Transformer
from unitunner.models.flux3.transformer import Flux3Transformer
from unitunner.models.image import ImageModel
from unitunner.models.krea2.transformer import Krea2Transformer
from unitunner.models.moss_image.transformer import MossImageTransformer
from unitunner.models.qwen_image.transformer import QwenImageTransformer
from unitunner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unitunner.models.z_image.transformer import ZImageTransformer
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
