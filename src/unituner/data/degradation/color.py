"""Apply torchvision-compatible color equations to CPU FP32 CHW RGB degradation pixels.

Adapted from torchvision.transforms._functional_tensor, BSD-3-Clause licensed.
Copyright (c) Soumith Chintala 2016. See licenses/TORCHVISION-LICENSE.txt.
Only floating three-channel tensors are supported; these are data transforms.
"""

import torch
from torch import Tensor


def rgb_to_grayscale(image: Tensor, num_output_channels: int = 1) -> Tensor:
    """Return weighted RGB luma as one or three channels, retaining FP32 arithmetic order."""
    red, green, blue = image.unbind(dim=-3)
    gray = (0.2989 * red + 0.587 * green + 0.114 * blue).unsqueeze(-3)
    return gray.expand_as(image) if num_output_channels == 3 else gray


def _blend(image: Tensor, reference: Tensor, factor: float) -> Tensor:
    return (factor * image + (1 - factor) * reference).clamp(0, 1)


def adjust_brightness(image: Tensor, factor: float) -> Tensor:
    """Blend against black and clamp to [0,1], without mutating the source."""
    return _blend(image, torch.zeros_like(image), factor)


def adjust_contrast(image: Tensor, factor: float) -> Tensor:
    """Blend against the spatial/channel mean of RGB luma, clamping to [0,1]."""
    return _blend(image, rgb_to_grayscale(image).mean(dim=(-3, -2, -1), keepdim=True), factor)


def adjust_saturation(image: Tensor, factor: float) -> Tensor:
    """Blend against per-pixel RGB luma, clamping to [0,1]."""
    return _blend(image, rgb_to_grayscale(image), factor)


def adjust_hue(image: Tensor, factor: float) -> Tensor:
    """Rotate hue by a fraction of a full turn, using the original FP32 RGB/HSV equations.

    Input is CPU FP32 CHW RGB in [0,1]; factor lies in [-0.5,0.5]. Gray/black
    pixels use safe denominators. Return new three-channel pixels without RNG.
    """
    red, green, blue = image.unbind(dim=-3)
    maximum, minimum = image.max(dim=-3).values, image.min(dim=-3).values
    equal = maximum == minimum
    difference = maximum - minimum
    ones = torch.ones_like(maximum)
    saturation = difference / torch.where(equal, ones, maximum)
    divisor = torch.where(equal, ones, difference)
    rc, gc, bc = ((maximum - channel) / divisor for channel in (red, green, blue))
    hr = (maximum == red) * (bc - gc)
    hg = ((maximum == green) & (maximum != red)) * (2.0 + rc - bc)
    hb = ((maximum != green) & (maximum != red)) * (4.0 + gc - rc)
    hue = (torch.fmod((hr + hg + hb) / 6.0 + 1.0, 1.0) + factor) % 1.0
    index = torch.floor(hue * 6.0)
    fraction = hue * 6.0 - index
    index = index.to(torch.int32) % 6
    p = (maximum * (1.0 - saturation)).clamp(0, 1)
    q = (maximum * (1.0 - saturation * fraction)).clamp(0, 1)
    t = (maximum * (1.0 - saturation * (1.0 - fraction))).clamp(0, 1)
    mask = index.unsqueeze(-3) == torch.arange(6, device=image.device).view(-1, 1, 1)
    channels = torch.stack(
        (
            torch.stack((maximum, q, p, p, t, maximum)),
            torch.stack((t, maximum, maximum, q, p, p)),
            torch.stack((p, p, t, maximum, maximum, q)),
        )
    )
    return torch.einsum("ijk,xijk->xjk", mask.to(image.dtype), channels)
