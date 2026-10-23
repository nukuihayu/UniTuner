# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Convert native Qwen2.1 RGB/RGBA media and encode image groups without sampling."""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

from unitunner.models.qwen_image.vae import QwenImageVAEExecution
from unitunner.models.qwen_image_2_1.vae import QwenImage21VAE

type ImageInput = Image.Image | NDArray[np.uint8]


def image_tensor(image: ImageInput, *, channels: int) -> Tensor:
    """Return CPU FP32 1C1HW pixels in [-1,1], preserving alpha for RGBA VAEs.

    Native Qwen2.1 media requires nonempty H/W divisible by32. HWC arrays must
    be uint8 RGB/RGBA; PIL modes convert explicitly. Inputs remain unchanged.
    The lower-level VAE accepts any image aligned to its actual spatial factor.
    """
    if channels not in (3, 4):
        raise ValueError("Qwen Image2.1 media requires RGB or RGBA vae channels")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError("Qwen Image2.1 arrays must be uint8 HWC RGB/RGBA")
        image = Image.fromarray(image)
    if min(image.size) < 32 or image.width % 32 or image.height % 32:
        raise ValueError("Qwen Image2.1 media H/W must be positive multiples of32")
    pixels = np.array(image.convert("RGBA" if channels == 4 else "RGB"), copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None].float() / 127.5 - 1


@torch.no_grad()
def encode_images(
    vae: QwenImage21VAE, images: Sequence[ImageInput], *, execution: QwenImageVAEExecution | None = None
) -> list[Tensor]:
    """Group equal-sized media, encode posterior means and return ordered C1HW latents.

    The caller owns model device/modes and optional slice/tile policy. Inputs are
    not mutated and no RNG draws occur with native dropout disabled/eval mode.
    All media geometry is validated before encoding the first group.
    """
    groups: dict[tuple[int, int], list[tuple[int, Tensor]]] = {}
    for index, image in enumerate(images):
        pixels = image_tensor(image, channels=vae.config.input_channels)
        if any(size % vae.config.spatial_factor for size in pixels.shape[-2:]):
            raise ValueError("Qwen Image2.1 media dimensions must align with vae spatial_factor")
        groups.setdefault((pixels.shape[-2], pixels.shape[-1]), []).append((index, pixels))
    device = next(vae.parameters()).device
    results: dict[int, Tensor] = {}
    for group in groups.values():
        latents = vae.encode(torch.cat([pixels for _, pixels in group]).to(device), execution=execution)
        for (index, _), latent in zip(group, latents, strict=True):
            results[index] = latent
    return [results[index] for index in range(len(images))]
