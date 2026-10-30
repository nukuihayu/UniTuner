"""Record the supported parameter modes and model-native restoration exclusions."""

from enum import StrEnum


class ModelFamily(StrEnum):
    """Canonical family identities independent of Python import paths."""

    Z_IMAGE = "z-image"
    MOSS_IMAGE = "moss-image"
    FLUX = "flux"
    FLUX2 = "flux2"
    FLUX3 = "flux3"
    QWEN_IMAGE = "qwen-image"
    QWEN_IMAGE_2_1 = "qwen-image-2-1"
    KREA2 = "krea2"
    WAN = "wan"
    MINIMAX_H3 = "minimax-h3"


def supports_restoration(family: ModelFamily, variant: str | None = None) -> bool:
    """Check native image restoration support, optionally for a selected variant."""
    if family in (ModelFamily.WAN, ModelFamily.MINIMAX_H3):
        return False
    if family == ModelFamily.FLUX:
        return variant not in ("fill", "kontext")
    if family == ModelFamily.MOSS_IMAGE:
        return variant != "edit"
    if family == ModelFamily.QWEN_IMAGE:
        return variant is None or variant == "original"
    return True
