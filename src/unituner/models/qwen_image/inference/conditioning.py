"""Encode Qwen's ordered VAE means and retain independently sized VL references."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from unituner.models.qwen_image.data.media import read_reference_image
from unituner.models.qwen_image.vae import QwenImageVAE


@dataclass(frozen=True, slots=True)
class QwenImageConditions:
    """Carry ordered CPU FP32 1C1HW normalized latents and owned HWC VL images."""

    latents: tuple[Tensor, ...]
    images: tuple[NDArray[np.uint8], ...]


@torch.no_grad()
def build_qwen_image_references(
    vae: QwenImageVAE,
    paths: Sequence[Path],
    *,
    resize: Literal["preferred", "original", "output"],
    resolution: tuple[int, int],
) -> QwenImageConditions:
    """Encode native posterior means, round through BF16 and publish CPU FP32.

    Retain independent source geometry and order, including original edit's
    additional VAE references (its VL context alone uses the first). Layered
    VAE input appends opaque alpha for RGB sources. No RNG, model modes or
    residency changes; the caller owns VAE lifetime and placement.
    """
    parameter = next(vae.parameters())
    latents, images = [], []
    with torch.autocast(parameter.device.type, enabled=False):
        for path in paths:
            image = read_reference_image(
                path,
                resize=resize,
                resolution=resolution,
                channels=vae.config.input_channels,
            )
            latent = vae.encode(image.pixels[None, :, None].to(parameter.device))
            latents.append(latent.to(torch.bfloat16).float().cpu())
            images.append(image.vision_pixels)
    return QwenImageConditions(tuple(latents), tuple(images))
