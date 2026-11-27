"""Decode aligned low-quality/high-quality images with aligned item-local crop and flip randomness."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageOps
from torch import Tensor

from unituner.data.datasets.restoration import AlignedImagePair
from unituner.data.degradation.config import DegradationConfig, ExtremeDegradationConfig
from unituner.data.degradation.pipeline import degrade_bsrgan_plus_native, degrade_pair, select_extreme_scale
from unituner.data.degradation.resize import resize_centered


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


def _tile_starts(length: int, size: int) -> list[int]:
    values = list(range(0, length - size + 1, size))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def _read_bsrgan_plus_patch(
    source: NDArray[np.uint8],
    high: NDArray[np.uint8],
    settings: ExtremeDegradationConfig,
    scale: int,
    *,
    crop_shape: tuple[int, int],
    seed: int,
    flip_probability: float,
) -> tuple[Tensor, Tensor]:
    height, width = crop_shape
    pad_y, pad_x = -high.shape[0] % scale, -high.shape[1] % scale
    if pad_y or pad_x:
        padding = ((0, pad_y), (0, pad_x), (0, 0))
        padded_high = np.pad(high, padding, mode="reflect")
        source = padded_high if source is high else np.pad(source, padding, mode="reflect")
        high = padded_high
    source_float = source.astype(np.float32) / np.float32(255)
    native = degrade_bsrgan_plus_native(source_float, settings, scale, seed, trace=None)
    del source_float
    low_height = (height + scale - 1) // scale
    low_width = (width + scale - 1) // scale
    generator = torch.Generator().manual_seed(seed)
    top = int(torch.randint(native.shape[0] - low_height + 1, (), generator=generator))
    left = int(torch.randint(native.shape[1] - low_width + 1, (), generator=generator))
    native_crop = native[top : top + low_height, left : left + low_width]
    aligned = resize_centered(native_crop, (low_height * scale, low_width * scale), "bicubic")
    low_pixels = aligned[:height, :width].clip(0, 1)
    low_pixels = (low_pixels * 255).round() / 255
    high_top, high_left = top * scale, left * scale
    high_pixels = high[high_top : high_top + height, high_left : high_left + width]
    low_tensor = torch.from_numpy(low_pixels).permute(2, 0, 1)
    high_tensor = torch.from_numpy(high_pixels.copy()).permute(2, 0, 1).float() / 255
    if float(torch.rand((), generator=generator)) < flip_probability:
        low_tensor, high_tensor = low_tensor.flip(-1), high_tensor.flip(-1)
    return low_tensor * 2 - 1, high_tensor * 2 - 1


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
    reflection padding. Random BSRGAN Plus patches crop native LQ after JPEG
    and delivery resizing. Tile calibration and other recipes degrade the HQ crop.
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
    if synthetic:
        if degradation is None or not degradation.enabled:
            raise ValueError("Missing low-quality image requires enabled synthetic degradation")
        extreme = degradation.extreme
        if sampling == "random" and extreme is not None and extreme.plus is not None:
            scale = select_extreme_scale(extreme, seed)
            if scale is not None:
                yield _read_bsrgan_plus_patch(
                    high,
                    high,
                    extreme,
                    scale,
                    crop_shape=(height, width),
                    seed=seed,
                    flip_probability=flip_probability,
                )
                return
    generator = torch.Generator().manual_seed(seed)
    if sampling == "random":
        if high.shape[:2] == (height, width):
            top, left = 0, 0
        else:
            top = int(torch.randint(high.shape[0] - height + 1, (), generator=generator))
            left = int(torch.randint(high.shape[1] - width + 1, (), generator=generator))
        positions = [(top, left)]
    else:
        positions = [
            (top, left) for top in _tile_starts(high.shape[0], height) for left in _tile_starts(high.shape[1], width)
        ]
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
            low_tensor, high_tensor = degrade_pair(high_tensor, degradation, generator.initial_seed())
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
