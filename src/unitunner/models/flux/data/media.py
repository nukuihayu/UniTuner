"""Validate native FLUX condition associations and aligned Fill repaint masks."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from unitunner.data.media.image import read_image, resize_pixels


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
    return read_fill_image(source, mask_path, resolution)


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

    Training and inference share uniform resize and center-crop geometry.
    Masks use nearest interpolation to retain labels. An explicit mask must
    match source dimensions; otherwise RGBA alpha is used only as the mask,
    never as RGB transparency. Zero normalized pixels before encoding.
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
        with mask:
            values = resize_pixels(np.array(mask.convert("RGB")), resolution, resample="mask")[..., 0].copy()
            values = torch.from_numpy(values).float().unsqueeze(0) / 255
    return read_image(source, resolution, rgb_before_resize=True) * (1 - values), values
