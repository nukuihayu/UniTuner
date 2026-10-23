"""Validate native FLUX condition associations and aligned Fill repaint masks."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from unitunner.data.media.image import read_image


def read_fill_condition(
    target: Path, source: Path, mask_path: Path | None, resolution: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """Return CPU normalized masked RGB CHW and 1HW repaint mask in [0,1].

    Source, target and explicit mask must have equal original sizes. Without an
    explicit mask, source must be RGBA and alpha=255 means repaint. RGB follows
    native bucket resize/crop; mask follows the identical geometry with nearest
    interpolation, preserving discrete labels. Pixels are zeroed after [-1,1]
    normalization, before VAE encoding. No EXIF rotation or RNG is implicit.
    """
    with Image.open(target) as high, Image.open(source) as image:
        if high.size != image.size:
            raise ValueError("Flux Fill target and condition require identical original dimensions")
        original = image.size
        if mask_path is None:
            if image.mode != "RGBA":
                raise ValueError("Flux Fill requires an explicit mask or RGBA condition")
            mask = image.getchannel("A")
        else:
            with Image.open(mask_path) as supplied:
                if supplied.size != original:
                    raise ValueError("Flux Fill mask dimensions differ from condition")
                mask = supplied.convert("L")
    with mask:
        width, height = original
        scale = max(resolution[0] / width, resolution[1] / height)
        resized_width, resized_height = int(width * scale + 0.5), int(height * scale + 0.5)
        left, top = (resized_width - resolution[0]) // 2, (resized_height - resolution[1]) // 2
        with mask.resize((resized_width, resized_height), Image.Resampling.NEAREST) as resized:
            with resized.crop((left, top, left + resolution[0], top + resolution[1])) as cropped:
                values = torch.from_numpy(np.array(cropped)).float().unsqueeze(0) / 255
    return read_image(source, resolution) * (1 - values), values


_KONTEXT_RESOLUTIONS = (
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
)


def read_reference_image(source: Path, *, preferred: bool) -> Tensor:
    """Return native CPU RGB CHW reference pixels with positive 16-pixel geometry.

    Preferred mode selects the published Kontext bucket by aspect ratio, then
    follows native RGB bucket resize/crop. Original mode crops right/bottom to
    multiples of sixteen without resizing. Reference size is independent of the
    generated target. No alpha compositing, EXIF rotation or RNG is implicit.
    """
    with Image.open(source) as image:
        width, height = image.size
        if preferred:
            _, width, height = min((abs(width / height - w / h), w, h) for w, h in _KONTEXT_RESOLUTIONS)
        else:
            width, height = width // 16 * 16, height // 16 * 16
            if min(width, height) < 16:
                raise ValueError("Flux Kontext reference must be at least sixteen pixels along each axis")
            with image.convert("RGB") as rgb, rgb.crop((0, 0, width, height)) as cropped:
                return torch.from_numpy(np.array(cropped)).permute(2, 0, 1).float() / 127.5 - 1
    return read_image(source, (width, height), rgb_before_resize=True)


def read_fill_image(source: Path, mask_path: Path | None, resolution: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Return native inference masked RGB CHW and 1HW white-repaint mask.

    Unlike training buckets, inference stretches source RGB with Lanczos and
    aligned masks with nearest-neighbor to the requested (width,height). An
    explicit mask must match source dimensions; otherwise RGBA alpha is used.
    Zero normalized pixels before encoding, retaining fractional repaint values.
    """
    with Image.open(source) as image:
        if mask_path is None:
            if image.mode != "RGBA":
                raise ValueError("Flux Fill requires an explicit mask or RGBA condition")
            mask = image.getchannel("A")
        else:
            with Image.open(mask_path) as supplied:
                if supplied.size != image.size:
                    raise ValueError("Flux Fill mask dimensions differ from condition")
                mask = supplied.convert("L")
        with mask, mask.resize(resolution, Image.Resampling.NEAREST) as resized_mask:
            values = torch.from_numpy(np.array(resized_mask)).float().unsqueeze(0) / 255
        with image.convert("RGB") as rgb, rgb.resize(resolution, Image.Resampling.LANCZOS) as resized:
            pixels = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 127.5 - 1
    return pixels * (1 - values), values
