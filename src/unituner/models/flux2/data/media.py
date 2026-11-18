"""Read FLUX.2 references using native aspect buckets and RGB resampling semantics."""

import math
from pathlib import Path

from PIL import Image
from torch import Tensor

from unituner.data.media.image import read_image


def reference_resolution(width: int, height: int, *, limit: int | None) -> tuple[int, int]:
    """Choose native16-aligned(width,height) under the original/reference area policy.

    Search the five neighboring width buckets and select the closest aspect ratio
    as in the old project. A single reference uses2024² pixels and multiple use
    1024²; original mode sets no area limit. Reject empty geometry before decoding
    or VAE allocation rather than deriving an invalid zero-sized image.
    """
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("FLUX.2 reference dimensions must be positive integers")
    if limit is not None and (type(limit) is not int or limit < 16):
        raise ValueError("FLUX.2 reference area side limit must be at least16")
    area = width * height if limit is None else min(width * height, limit * limit)
    ratio = width / height
    base = int(math.sqrt(area * ratio)) // 16 * 16
    fallback_height = int(math.sqrt(area / ratio)) // 16 * 16
    best = None
    difference = math.inf
    for offset in range(-2, 3):
        candidate_width = base + offset * 16
        if candidate_width <= 0:
            continue
        candidate_height = (area // candidate_width) // 16 * 16
        if candidate_height <= 0:
            continue
        error = abs(candidate_width / candidate_height - ratio)
        if error < difference:
            best = (candidate_width, candidate_height)
            difference = error
    result = (base, fallback_height) if best is None else best
    if min(result) < 16:
        raise ValueError("FLUX.2 reference cannot form a nonempty16-aligned aspect bucket")
    return result


def read_reference_image(path: Path, *, count: int, original: bool = False) -> Tensor:
    """Return CPU FP32 RGB CHW[-1,1] pixels at the native aspect-preserving bucket.

    Drop alpha before resampling, use Lanczos and half-up center crop.
    Preserve historical orientation without EXIF rotation. Reference count chooses
    the single/multiple area limit; explicit original mode preserves source area
    before stride alignment. Files are closed and no RNG is consumed.
    """
    if type(count) is not int or count < 1:
        raise ValueError("FLUX.2 reference count must be positive")
    with Image.open(path) as image:
        resolution = reference_resolution(*image.size, limit=None if original else 2024 if count == 1 else 1024)
    return read_image(path, resolution, rgb_before_resize=True)
