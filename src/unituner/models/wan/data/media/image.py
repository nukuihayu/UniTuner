"""Read native Wan inference image geometry and distinct CLIP/VAE pixels."""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor


@dataclass(frozen=True, slots=True)
class WanImagePixels:
    """Own CPU FP32 original/resized RGB CHW pixels and effective width/height."""

    original: Tensor
    resized: Tensor
    resolution: tuple[int, int]


def read_wan_image(path: Path, max_resolution: tuple[int, int]) -> WanImagePixels:
    """Match native I2V area-budget geometry and torchvision normalization order.

    Preserve aspect ratio using floor-to16 latent patch geometry. CLIP receives
    the original RGB image; VAE pixels use OpenCV area when shrinking height,
    otherwise cubic, with no center crop. Divide uint8 by255, subtract0.5 then
    divide0.5 separately; do not replace this with the cache's /127.5-1 formula.
    Reject a budget producing zero patch dimensions before encoder/model IO.
    """
    if min(max_resolution) < 16 or any(size % 16 for size in max_resolution):
        raise ValueError("Wan inference image budget requires16-aligned dimensions")
    with Image.open(path) as source:
        with source.convert("RGB") as rgb:
            raw = np.array(rgb)
    aspect = raw.shape[0] / raw.shape[1]
    area = math.prod(max_resolution)
    height = round(math.sqrt(area * aspect) // 8 // 2 * 2) * 8
    width = round(math.sqrt(area / aspect) // 8 // 2 * 2) * 8
    if min(height, width) < 16:
        raise ValueError("Wan image area budget produces no complete16-pixel patch for this aspect ratio")
    resized = cv2.resize(
        raw, (width, height), interpolation=cv2.INTER_AREA if height < raw.shape[0] else cv2.INTER_CUBIC
    )
    original = torch.from_numpy(raw).permute(2, 0, 1).contiguous().float().div_(255).sub_(0.5).div_(0.5)
    pixels = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float().div_(255).sub_(0.5).div_(0.5)
    return WanImagePixels(original, pixels, (width, height))
