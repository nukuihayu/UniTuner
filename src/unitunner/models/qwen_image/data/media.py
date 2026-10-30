"""Read independent Qwen VAE/VL reference buckets and native packed edit masks."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from unitunner.data.media.image import read_image


@dataclass(frozen=True, slots=True)
class QwenImageReference:
    """Carry owned CPU CHW[-1,1] VAE pixels and uint8 HWC RGB(A) VL pixels."""

    pixels: Tensor
    vision_pixels: NDArray[np.uint8]


def reference_resolution(width: int, height: int, *, area: int, alignment: int = 16) -> tuple[int, int]:
    """Choose a native aligned aspect bucket at the supplied positive area.

    Search five neighboring width buckets; ties retain the first candidate.
    Official VAE/VL areas are1024²/384², original mode uses source area.
    Alignment defaults to16; Qwen2.1 uses32. Reject empty aligned buckets.
    """
    if min(width, height, area, alignment) < 1:
        raise ValueError("Qwen reference size and bucket area must be positive")
    ratio = width / height
    base = int(math.sqrt(area * ratio)) // alignment * alignment
    best = None
    error = math.inf
    for offset in range(-2, 3):
        candidate_width = base + offset * alignment
        if candidate_width <= 0:
            continue
        candidate_height = (area // candidate_width) // alignment * alignment
        if candidate_height <= 0:
            continue
        difference = abs(candidate_width / candidate_height - ratio)
        if difference < error:
            best, error = (candidate_width, candidate_height), difference
    if best is None:
        raise ValueError(f"Qwen reference cannot form a nonempty{alignment}-aligned aspect bucket")
    return best


def read_reference_image(
    path: Path,
    *,
    resize: Literal["preferred", "original", "output"],
    resolution: tuple[int, int],
    channels: Literal[3, 4],
) -> QwenImageReference:
    """Read native independent VAE/VL buckets without EXIF rotation or alpha compositing.

    Convert to RGBA before resize, then drop alpha only for RGB consumers. Output
    mode uses the explicit(width,height) for both consumers; original uses source
    area buckets; preferred uses1024²/384². Native Lanczos, half-up
    resize and center crop match the image boundary. No input mutation or RNG.
    """
    if resize not in ("preferred", "original", "output") or channels not in (3, 4):
        raise ValueError("Qwen references require preferred/original/output resize and RGB/RGBA channels")
    if min(resolution) < 16 or any(value % 16 for value in resolution):
        raise ValueError("Qwen output reference size must be16-aligned")
    with Image.open(path) as image:
        width, height = image.size
    if resize == "output":
        vae_size = vision_size = resolution
    else:
        area = width * height
        vae_size = reference_resolution(width, height, area=1024**2 if resize == "preferred" else area)
        vision_size = reference_resolution(width, height, area=384**2 if resize == "preferred" else area)
    pixels = read_image(path, vae_size, channels=4)[:channels]
    vision = read_image(path, vision_size, channels=4)[:channels]
    array = ((vision + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    return QwenImageReference(pixels.contiguous(), array)


def read_inpainting_mask(path: Path, resolution: tuple[int, int]) -> Tensor:
    """Read native first RGB channel to CPU FP32 1N1 repaint weights, rounded through BF16.

    Resize/crop at target pixel geometry, bilinearly downsample to the packed
    H/16,W/16 grid with align_corners=False, then map[-1,1] to[0,1]. White means
    regenerate. Do not substitute luminance conversion or nearest interpolation.
    """
    pixels = read_image(path, resolution, channels=4)[:1].unsqueeze(0)
    pixels = F.interpolate(
        pixels, size=(resolution[1] // 16, resolution[0] // 16), mode="bilinear", align_corners=False
    )
    return ((pixels + 1.0) / 2.0).reshape(1, -1, 1).to(torch.bfloat16).float()
