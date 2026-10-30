"""Build shared RGBA reference pixels, native Qwen2.1 prompts and unrounded repaint masks."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from unitunner.config.sampling import SamplingConfig
from unitunner.data.media.image import resize_pixels
from unitunner.models.qwen_image_2_1.conditioning import QwenImage21TextFeatures
from unitunner.models.qwen_image_2_1.data.encoding import encode_images
from unitunner.models.qwen_image_2_1.text.encoder import QwenImage21TextEncoder
from unitunner.models.qwen_image_2_1.vae import QwenImage21VAE


@dataclass(frozen=True, slots=True)
class QwenImage21Conditions:
    """Own ordered CPU1C1HW means, prompt-specific slots and optional FP32 1N1 mask."""

    positive: QwenImage21TextFeatures
    negative: QwenImage21TextFeatures | None
    references: tuple[Tensor, ...]
    mask: Tensor | None


def reference_resolution(width: int, height: int, *, area: int) -> tuple[int, int]:
    """Choose the native32-aligned aspect bucket using five candidates, retaining first ties."""
    if min(width, height, area) < 1:
        raise ValueError("Qwen Image2.1 reference dimensions and area must be positive")
    ratio = width / height
    base = int(math.sqrt(area * ratio)) // 32 * 32
    best = None
    error = math.inf
    for offset in range(-2, 3):
        candidate_width = base + offset * 32
        if candidate_width <= 0:
            continue
        candidate_height = (area // candidate_width) // 32 * 32
        if candidate_height <= 0:
            continue
        difference = abs(candidate_width / candidate_height - ratio)
        if difference < error:
            best, error = (candidate_width, candidate_height), difference
    if best is None:
        raise ValueError("Qwen Image2.1 reference cannot form a nonempty32-aligned bucket")
    return best


@torch.no_grad()
def build_qwen_image_21_conditions(
    vae: QwenImage21VAE, text: QwenImage21TextEncoder, sampling: SamplingConfig
) -> QwenImage21Conditions:
    """Encode both consumers from identical RGBA reference bytes, without mode/residency mutation.

    Preferred reference area is1024²; original preserves source area before32
    alignment; output uses exact requested dimensions. Resizing keeps native
    alpha-first Lanczos center crop, with no EXIF rotation. Mask
    uses first RGB channel/255 then bilinear latent-grid interpolation, without
    BF16 rounding. Caller owns frozen FP32 encoders and their selected devices.
    """
    if vae.config.spatial_factor != 16 or vae.config.latent_channels != 64:
        raise ValueError("Qwen Image2.1 generation requires vae factor16 and latent64")
    images = []
    for name in sampling.reference_images:
        with Image.open(name) as image:
            rgba = np.array(image.convert("RGBA"), copy=True)
            width, height = image.size
        size = (
            (sampling.width, sampling.height)
            if sampling.conditioning_resize == "output"
            else reference_resolution(
                width, height, area=width * height if sampling.conditioning_resize == "original" else 1024**2
            )
        )
        images.append(resize_pixels(rgba, size))
    positive = text.encode([sampling.prompt], references=[images])[0]
    negative = None
    if sampling.guidance is not None and sampling.guidance > 1:
        negative = text.encode(
            ["" if sampling.negative_prompt is None else sampling.negative_prompt], references=[images]
        )[0]
    references = tuple(value[None] for value in encode_images(vae, images))
    mask = None
    if sampling.mask_image is not None:
        if not references or references[0].shape[-2:] != (sampling.height // 16, sampling.width // 16):
            raise ValueError("Qwen Image2.1 inpainting first reference must match output dimensions")
        with Image.open(Path(sampling.mask_image)) as image:
            values = resize_pixels(np.array(image.convert("RGB"), copy=True), (sampling.width, sampling.height))[
                :, :, 0
            ].copy()
        pixels = torch.from_numpy(values).float()[None, None] / 255
        mask = (
            F.interpolate(
                pixels, size=(sampling.height // 16, sampling.width // 16), mode="bilinear", align_corners=False
            )
            .flatten(2)
            .transpose(1, 2)
        )
    return QwenImage21Conditions(positive, negative, references, mask)
