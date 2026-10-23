"""Decode aligned low-quality/high-quality images with aligned item-local crop and flip randomness."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageOps
from torch import Tensor

from unitunner.data.datasets.restoration import AlignedImagePair
from unitunner.data.degradation.config import DegradationConfig
from unitunner.data.degradation.pipeline import degrade_image


def _read_rgb(path: Path) -> NDArray[np.uint8]:
    """Own one RGB array without extra full-image PIL orientation/conversion copies."""
    with Image.open(path) as image:
        ImageOps.exif_transpose(image, in_place=True)
        if image.mode == "RGB":
            return np.array(image, copy=True)
        with image.convert("RGB") as rgb:
            return np.array(rgb, copy=True)


def read_restoration_pair(
    pair: AlignedImagePair, resolution: tuple[int, int], seed: int, flip_probability: float
) -> tuple[Tensor, Tensor]:
    """Return owned FP32 RGB CHW LQ/HQ [-1,1] using the shared restoration crop policy.

    Apply EXIF orientation, require equal aligned image dimensions, reflect-pad bottom
    and right, then choose a shared crop/flip from a local CPU generator. Resolution
    is (width,height). No resampling or global RNG changes occur.
    """
    return read_aligned_images(
        pair.low,
        pair.high,
        resolution=resolution,
        seed=seed,
        flip_probability=flip_probability,
        degradation=pair.degradation,
    )


def read_aligned_images(
    low_path: Path | None,
    high_path: Path,
    *,
    resolution: tuple[int, int],
    seed: int,
    flip_probability: float = 0.0,
    degradation: DegradationConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Read aligned RGB CHW [-1,1] pairs for training or VAE-only calibration.

    Resolution is pixel (W,H). Shared EXIF, reflection padding, crop, flip and
    optional HQ-derived degradation use local generators. No captions/cache,
    transformer, global RNG or resampling of real aligned pairs are involved.
    """
    return next(
        iter_aligned_images(
            low_path,
            high_path,
            resolution=resolution,
            seed=seed,
            flip_probability=flip_probability,
            degradation=degradation,
        )
    )


def iter_aligned_images(
    low_path: Path | None,
    high_path: Path,
    *,
    resolution: tuple[int, int],
    seed: int,
    flip_probability: float = 0.0,
    degradation: DegradationConfig | None = None,
    sampling: Literal["random", "tiles"] = "random",
) -> Iterator[tuple[Tensor, Tensor]]:
    """Decode once, then yield a training crop or row-major full-resolution tiles.

    Tiles touch the final edge with overlap when necessary; small images use
    reflection padding. Degradation runs after cropping, as in training. Every
    tile has independent local augmentation randomness; no global RNG is used.
    """
    if sampling not in ("random", "tiles"):
        raise ValueError("Calibration sampling requires random or tiles")
    arrays = []
    for path in (high_path,) if low_path is None else (low_path, high_path):
        arrays.append(_read_rgb(path))
    synthetic = low_path is None
    low, high = (arrays[0], arrays[0]) if synthetic else arrays
    del arrays
    if low.shape != high.shape:
        raise ValueError("Aligned LQ/HQ images must have equal sizes after EXIF orientation")
    width, height = resolution
    if min(width, height) < 1 or not 0 <= flip_probability <= 1:
        raise ValueError("Restoration crop requires positive geometry and flip probability in [0,1]")
    padding = ((0, max(0, height - high.shape[0])), (0, max(0, width - high.shape[1])), (0, 0))
    if padding[0][1] or padding[1][1]:
        high = np.pad(high, padding, mode="reflect")
        low = high if synthetic else np.pad(low, padding, mode="reflect")
    generator = torch.Generator().manual_seed(seed)
    if sampling == "random":
        if high.shape[:2] == (height, width):
            top, left = 0, 0
        else:
            top = int(torch.randint(high.shape[0] - height + 1, (), generator=generator))
            left = int(torch.randint(high.shape[1] - width + 1, (), generator=generator))
        positions = [(top, left)]
    else:

        def starts(length: int, size: int) -> list[int]:
            values = list(range(0, length - size + 1, size))
            if values[-1] != length - size:
                values.append(length - size)
            return values

        positions = [(top, left) for top in starts(high.shape[0], height) for left in starts(high.shape[1], width)]
    for index, (top, left) in enumerate(positions):
        if sampling == "tiles":
            generator.manual_seed(seed + index * 1_000_003)
        flip = float(torch.rand((), generator=generator)) < flip_probability
        tensors = []
        for value in (high,) if synthetic else (low, high):
            pixels = (
                torch.from_numpy(value[top : top + height, left : left + width].copy()).permute(2, 0, 1).float() / 255
            )
            tensors.append(pixels.flip(-1) if flip else pixels)
        if synthetic:
            if degradation is None or not degradation.enabled:
                raise ValueError("Missing low-quality image requires enabled synthetic degradation")
            high_tensor = tensors[0]
            low_tensor = degrade_image(high_tensor, degradation, generator.initial_seed())
        else:
            low_tensor, high_tensor = tensors
        yield low_tensor * 2 - 1, high_tensor * 2 - 1
        del tensors, pixels, low_tensor, high_tensor


def read_restoration_image(path: Path, *, granularity: int) -> tuple[Tensor, tuple[int, int]]:
    """Decode RGB FP32 CHW [-1,1] with EXIF orientation and bottom/right reflection padding.

    Return independent CPU storage and original oriented (width,height). Pad to
    a multiple of positive granularity without resize/crop or random operations.
    NumPy reflection supports padding larger than tiny input dimensions.
    """
    if granularity < 1:
        raise ValueError("Restoration image granularity must be positive")
    pixels = _read_rgb(path)
    height, width = pixels.shape[:2]
    if height % granularity or width % granularity:
        pixels = np.pad(pixels, ((0, -height % granularity), (0, -width % granularity), (0, 0)), mode="reflect")
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().float() / 127.5 - 1, (width, height)
