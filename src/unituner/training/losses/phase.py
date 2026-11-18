"""Constrain periodic residuals while protecting affine lighting and target texture."""

import math

import torch
from torch import Tensor


def phase_loss(
    generated: Tensor,
    target: Tensor,
    *,
    period: int = 16,
    cycles: int = 8,
    epsilon: float = 1e-3,
) -> Tensor:
    """Penalize periodic errors after fitting local lighting and target contrast.

    RGB inputs use BCHW and the restoration [-1, 1] scale. Each window contains
    ``cycles`` complete periods per axis. First remove a plane from both the
    error and target. Remove the error component proportional to that detrended
    target, using ridge variance 1e-6. Apply Charbonnier to the remaining
    zero-mean phase template. Statistics stay FP32; targets are detached.

    Interior windows do not overlap. A final full window touches each remaining
    edge, overlapping its neighbor without padding or discarding pixels. Every
    window has equal weight. Images must fit at least one full window.

    This supplementary constraint cannot supervise brightness or target contrast
    on its own. Its target-dependent linear operator is not a single orthogonal
    projector. Gradients differentiate the scalar objective normally, including
    the fitted contrast coefficient; no gradient filtering is used.
    """
    if (
        generated.ndim != 4
        or generated.shape[1] != 3
        or generated.shape != target.shape
        or not generated.numel()
        or generated.device != target.device
        or not generated.is_floating_point()
        or not target.is_floating_point()
    ):
        raise ValueError("Phase loss requires matching nonempty floating RGB BCHW images/device")
    if period < 2 or cycles < 2 or not math.isfinite(epsilon) or not 1e-12 <= epsilon <= 1:
        raise ValueError("Phase loss requires period/cycles >= 2 and finite epsilon in [1e-12, 1]")
    size = period * cycles
    height, width = generated.shape[-2:]
    if min(height, width) < size:
        raise ValueError(
            f"Phase loss window {size} exceeds image {height}x{width}; reduce phase_period or phase_cycles"
        )
    with torch.autocast(generated.device.type, enabled=False):
        reference = target.detach().float()
        error = generated.float() - reference
        error_tiles = _windows(error, size)
        target_tiles = _windows(reference, size)
        coordinate = torch.arange(size, device=error.device, dtype=torch.float32) - (size - 1) / 2
        variance = (size * size - 1) / 12
        error_tiles = _detrend(error_tiles, coordinate, variance)
        target_tiles = _detrend(target_tiles, coordinate, variance)
        # A variance ridge keeps constant/near-constant targets well conditioned.
        coefficient = (error_tiles * target_tiles).mean((-2, -1), keepdim=True) / (
            target_tiles.square().mean((-2, -1), keepdim=True) + 1e-6
        )
        residual = error_tiles - coefficient * target_tiles
        means = residual.reshape(*residual.shape[:-2], cycles, period, cycles, period).mean((-4, -2))
        means = means - means.mean((-2, -1), keepdim=True)
        penalty = means.square() / (torch.sqrt(means.square() + epsilon * epsilon) + epsilon)
        return penalty.mean((-2, -1)).mean()


def _windows(value: Tensor, size: int) -> Tensor:
    """Cover the full image with complete windows and overlapping edge windows."""
    height, width = value.shape[-2:]
    rows = value.unfold(2, size, size)
    if height % size:
        rows = torch.cat((rows, value[:, :, -size:, :].transpose(-1, -2).unsqueeze(2)), dim=2)
    tiles = rows.unfold(3, size, size)
    if width % size:
        tiles = torch.cat((tiles, rows[:, :, :, -size:, :].transpose(-1, -2).unsqueeze(3)), dim=3)
    return tiles


def _detrend(tiles: Tensor, coordinate: Tensor, variance: float) -> Tensor:
    """Remove the fitted constant and two spatial slopes from each window/channel."""
    slope_x = (tiles.mean(-2) * coordinate).mean(-1) / variance
    slope_y = (tiles.mean(-1) * coordinate).mean(-1) / variance
    return (
        tiles
        - tiles.mean((-2, -1), keepdim=True)
        - slope_x[..., None, None] * coordinate[None, :]
        - slope_y[..., None, None] * coordinate[:, None]
    )
