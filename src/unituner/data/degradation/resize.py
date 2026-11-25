"""Center-aligned resizing on the actual integer output grid."""

import math

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F


def scaled_shape(shape: tuple[int, int], factor: float, *, multiple: int = 1) -> tuple[int, int]:
    """Round to the nearest valid shape, with half ties rounding up."""
    height, width = shape
    return (
        max(1, math.floor(height * factor / multiple + 0.5)) * multiple,
        max(1, math.floor(width * factor / multiple + 0.5)) * multiple,
    )


def _area_axis(image: torch.Tensor, length: int, axis: int) -> torch.Tensor:
    """Average exact overlaps between input pixels and output pixel footprints."""
    image = image.movedim(axis, -1)
    width = image.shape[-1]
    if width == length:
        return image.movedim(-1, axis)
    numerator = torch.arange(length + 1, device=image.device) * width
    index, remainder = numerator // length, numerator % length
    integral = F.pad(image.cumsum(-1), (1, 0))
    edges = integral[..., index] + image[..., index.clamp_max(width - 1)] * (remainder.double() / length)
    return (edges.diff(dim=-1) * (length / width)).movedim(-1, axis)


def _nearest_axis(image: torch.Tensor, length: int, axis: int) -> torch.Tensor:
    """Use the nearest pixel center, averaging exact ties to retain reflection symmetry."""
    width = image.shape[axis]
    if width == length:
        return image
    # The source edge coordinate is (j + 0.5) * width / length.
    # An integer edge coordinate is exactly halfway between two pixel centers.
    numerator = (2 * torch.arange(length, device=image.device) + 1) * width
    index, remainder = numerator // (2 * length), numerator % (2 * length)
    right = image.index_select(axis, index)
    left = image.index_select(axis, index - (remainder == 0).long())
    return (left + right) * 0.5


def resize_centered[T: (np.float32, np.uint8)](image: NDArray[T], shape: tuple[int, int], mode: str) -> NDArray[T]:
    """Resample with common pixel centers and exact integer output dimensions."""
    if image.shape[:2] == shape:
        return image
    value = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
    if mode == "area":
        output = _area_axis(_area_axis(value.double(), shape[1], -1), shape[0], -2)
    elif mode == "nearest":
        output = _nearest_axis(_nearest_axis(value.float(), shape[1], -1), shape[0], -2)
    else:
        output = F.interpolate(value.double(), size=shape, mode=mode, align_corners=False, antialias=False)
    if value.dtype == torch.uint8:
        output = output.round().clamp(0, 255)
    return output.to(value.dtype)[0].permute(1, 2, 0).contiguous().numpy()
