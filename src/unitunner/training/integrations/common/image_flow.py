"""Describe normalized image-flow requests independently of native velocity/time conventions."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class ImageFlowRequest:
    """Carry BCHW clean/noise, B noise fractions and BLD captions with optional BL mask.

    Latents are already normalized in model units. Sigma is a noise fraction in
    [0,1]; each binding converts it to native time and chooses the velocity sign.
    """

    latents: Tensor
    noise: Tensor
    sigma: Tensor
    captions: Tensor
    caption_mask: Tensor | None = None
    references: tuple[Tensor, ...] = ()


@dataclass(frozen=True, slots=True)
class ImageFlowPrediction:
    """Carry native BCHW velocity/target and B native timesteps without detaching.

    Z-Image uses clean-minus-noise and 1-sigma time. MOSS uses noise-minus-clean
    and sigma*1000 time. Loss compares native values; no sign is inferred here.
    """

    prediction: Tensor
    target: Tensor
    timesteps: Tensor
