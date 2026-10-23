"""Runtime type contract for the eight native image denoisers."""

from unitunner.models.flux.transformer import FluxTransformer
from unitunner.models.flux2.transformer import Flux2Transformer
from unitunner.models.flux3.transformer import Flux3Transformer
from unitunner.models.krea2.transformer import Krea2Transformer
from unitunner.models.moss_image.transformer import MossImageTransformer
from unitunner.models.qwen_image.transformer import QwenImageTransformer
from unitunner.models.qwen_image_2_1.transformer import QwenImage21Transformer
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
