"""Encode independent Krea2 CFG branches with compact selected-layer text features."""

from unituner.config.sampling import SamplingConfig
from unituner.models.krea2.conditioning import Krea2TextFeatures, batch_text
from unituner.models.krea2.text.encoder import Krea2TextEncoder


def encode_krea2_prompts(
    text: Krea2TextEncoder, sampling: SamplingConfig, *, variant: str = "raw"
) -> tuple[Krea2TextFeatures, Krea2TextFeatures | None]:
    """Encode positive and, for CFG above1, negative text with independent padding.

    Features stay on the text owner's device. The caller owns evaluation mode
    and text lifetime. An omitted negative prompt uses the native empty string.
    """
    positive = batch_text(text.encode([sampling.prompt]))
    negative = (
        batch_text(text.encode(["" if sampling.negative_prompt is None else sampling.negative_prompt]))
        if (sampling.guidance is None and variant != "turbo")
        or (sampling.guidance is not None and sampling.guidance > 1)
        else None
    )
    return positive, negative
