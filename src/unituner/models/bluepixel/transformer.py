"""Distinguish BluePixel's editing checkpoint from ordinary FLUX.2 Klein."""

from typing import ClassVar

from unituner.models.catalog import ModelFamily
from unituner.models.flux2.transformer import Flux2Transformer


class BluePixelTransformer(Flux2Transformer):
    """Reuse the native Klein-9B topology with BluePixel-owned sampling semantics."""

    family: ClassVar[ModelFamily] = ModelFamily.BLUEPIXEL
