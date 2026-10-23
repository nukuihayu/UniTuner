"""Compute normalized Lab histogram CDF distance with sparse bin deposition."""

import torch
from torch.nn import functional as F


def _normalized_lab(image: torch.Tensor) -> torch.Tensor:
    """Convert RGB BCHW [-1,1] to D65 Lab channels normalized to [0,1]."""
    rgb = ((image.float() + 1) * 0.5).clamp(0, 1)
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)
    transform = linear.new_tensor(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
    )
    xyz = torch.einsum("ij,bjhw->bihw", transform, linear)
    xyz = xyz / xyz.new_tensor([0.95047, 1.0, 1.08883])[None, :, None, None]
    delta = 6 / 29
    transformed = torch.where(xyz > delta**3, xyz.clamp_min(1e-12).pow(1 / 3), xyz / (3 * delta**2) + 4 / 29)
    x, y, z = transformed.unbind(1)
    lab = torch.stack((116 * y - 16, 500 * (x - y), 200 * (y - z)), dim=1)
    offset = lab.new_tensor([0.0, 128.0, 128.0])[None, :, None, None]
    scale = lab.new_tensor([100.0, 255.0, 255.0])[None, :, None, None]
    return ((lab + offset) / scale).clamp(0, 1)


def _soft_histogram(lab: torch.Tensor, bins: int) -> torch.Tensor:
    """Deposit each Lab sample in its adjacent centers without a pixel/bin volume."""
    positions = lab.flatten(2) * (bins - 1)
    lower = positions.floor().long()
    upper = (lower + 1).clamp_max(bins - 1)
    fraction = positions - lower
    histogram = positions.new_zeros((*positions.shape[:2], bins))
    histogram.scatter_add_(2, lower, 1 - fraction)
    histogram.scatter_add_(2, upper, fraction)
    return histogram / histogram.sum(-1, keepdim=True).clamp_min(1e-8)


def histogram_loss(generated: torch.Tensor, target: torch.Tensor, bins: int = 32) -> torch.Tensor:
    """Compute per-image Lab soft-histogram cumulative distance.

    Both images are converted from sRGB to D65 Lab, normalized to [0,1] and
    bilinearly reduced to a maximum side of 256. Triangular bin interpolation
    and mean absolute CDF difference preserve the original HistogramLoss formula; this is a
    dimensionless CDF average, not RGB Wasserstein distance in intensity units.

    Args:
        generated: Floating RGB BCHW image in [-1,1], differentiable to G.
            Decoder overshoot is clamped for this loss only.
        target: Same-shaped image on the same device; detached internally.
        bins: Number of uniformly spaced Lab centers, at least two; default 32.

    Returns:
        Scalar FP32 distance averaged over images, Lab channels and CDF bins.
        Storage scales with pixels plus bins, avoiding a dense pixel/bin tensor.

    Raises:
        ValueError: Images do not share nonempty RGB BCHW shapes/device or bins < 2.
    """
    if (
        bins < 2
        or generated.shape != target.shape
        or generated.ndim != 4
        or generated.shape[1] != 3
        or not generated.numel()
        or generated.device != target.device
    ):
        raise ValueError("Histogram loss requires matching nonempty RGB BCHW images/device and at least two bins")
    with torch.autocast(generated.device.type, enabled=False):
        predicted_lab, target_lab = _normalized_lab(generated), _normalized_lab(target.detach())
        height, width = generated.shape[-2:]
        if max(height, width) > 256:
            scale = 256 / max(height, width)
            size = (max(1, round(height * scale)), max(1, round(width * scale)))
            predicted_lab, target_lab = (
                F.interpolate(value, size=size, mode="bilinear", align_corners=False)
                for value in (predicted_lab, target_lab)
            )
        predicted = _soft_histogram(predicted_lab, bins)
        reference = _soft_histogram(target_lab, bins)
        return (predicted.cumsum(-1) - reference.cumsum(-1)).abs().mean()
