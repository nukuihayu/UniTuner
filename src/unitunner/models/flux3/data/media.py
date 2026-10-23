# Derived from Black Forest Labs FLUX3 Image Klein, Apache-2.0.
"""Read FLUX3 references with official EXIF/RGB, padding and multi-reference crop rules."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor


@dataclass(frozen=True, slots=True)
class Flux3Reference:
    """Carry CPU FP32 CHW [-1,1] pixels and an optional unpadded (left,top,right,bottom) box."""

    pixels: Tensor
    crop_box: tuple[int, int, int, int] | None = None


def reference_canvas(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return official single-reference fitted size and 16-aligned canvas, each (width,height).

    Preserve aspect ratio under four megapixels and 4096 pixels per side, without
    upscaling fitted pixels; the padded canvas is at least 64 pixels per side.
    """
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("FLUX3 reference dimensions must be positive integers")
    scale = min(1.0, math.sqrt(4 * 1024 * 1024 / (width * height)), 4096 / max(width, height))
    while True:
        fit = max(1, round(width * scale)), max(1, round(height * scale))
        canvas = tuple(min(4096, max(64, -(-value // 16) * 16)) for value in fit)
        if canvas[0] * canvas[1] <= 4 * 1024 * 1024:
            return fit, (canvas[0], canvas[1])
        scale *= 0.995


def read_reference_images(sources: Sequence[Path]) -> tuple[Flux3Reference, ...]:
    """Return zero to ten official references without mutating files or consuming RNG.

    Apply EXIF transpose then discard alpha before Lanczos resize. A single image
    is centered and reflect padded (symmetric for sides <=16) to its canvas;
    crop_box records the fitted image for optional post-decode cropping. Multiple
    images are each fit under one megapixel then center cropped to a 16-pixel grid.
    Reject a multi-reference side below sixteen after fitting: the official crop
    would otherwise yield a zero-width/height image and fail later in the VAE.
    """
    if len(sources) > 10:
        raise ValueError("FLUX3 supports at most ten reference images")
    results = []
    for source in sources:
        with Image.open(source) as opened, ImageOps.exif_transpose(opened) as oriented, oriented.convert("RGB") as rgb:
            crop_box = None
            if len(sources) == 1:
                fit, (width, height) = reference_canvas(*rgb.size)
                with rgb.resize(fit, Image.Resampling.LANCZOS) as fitted:
                    array = np.array(fitted)
                if fit != (width, height):
                    left, top = (width - fit[0]) // 2, (height - fit[1]) // 2
                    padding = ((top, height - fit[1] - top), (left, width - fit[0] - left), (0, 0))
                    array = np.pad(array, padding, mode="reflect" if min(fit) > 16 else "symmetric")
                    crop_box = (left, top, left + fit[0], top + fit[1])
            else:
                width, height = rgb.size
                if width * height > 1024 * 1024:
                    scale = math.sqrt(1024 * 1024 / (width * height))
                    width, height = int(width * scale), int(height * scale)
                if min(width, height) < 16:
                    raise ValueError(
                        "FLUX3 multi-reference images require sides >=16 after fitting under one megapixel"
                    )
                with rgb.resize((width, height), Image.Resampling.LANCZOS) as fitted:
                    left, top = (width % 16) // 2, (height % 16) // 2
                    with fitted.crop((left, top, left + width // 16 * 16, top + height // 16 * 16)) as cropped:
                        array = np.array(cropped)
            pixels = torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1.0
            results.append(Flux3Reference(pixels, crop_box))
    return tuple(results)
