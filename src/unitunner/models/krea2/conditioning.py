"""Batch selected-layer Krea2 captions and preserve image-first spatial patch order."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class Krea2TextFeatures:
    """Borrow padded BSLD selected-layer text and host valid-prefix lengths."""

    features: Tensor
    lengths: tuple[int, ...]


def batch_text(rows: tuple[Tensor, ...]) -> Krea2TextFeatures:
    """Pad nonempty SLD captions to batch maximum while preserving every graph.

    Rows contain compacted valid tokens only, in encoder order. Host lengths are
    derived before padding; no device-to-host reads or RNG operations occur.
    """
    if not rows or any(row.ndim != 3 or min(row.shape) < 1 or not row.is_floating_point() for row in rows):
        raise ValueError("Krea2 captions require nonempty floating SLD rows")
    first = rows[0]
    if any(row.shape[1:] != first.shape[1:] or row.dtype != first.dtype or row.device != first.device for row in rows):
        raise ValueError("Krea2 caption layers/width/dtype/device must agree")
    lengths = tuple(row.shape[0] for row in rows)
    return Krea2TextFeatures(
        torch.stack([F.pad(row, (0, 0, 0, 0, 0, max(lengths) - row.shape[0])) for row in rows]), lengths
    )


def pack_latents(latents: Tensor, *, patch: int) -> Tensor:
    """View floating BCHW as B(H/P*W/P)(C*P*P) in native channel/row/column order."""
    if (
        latents.ndim != 4
        or min(latents.shape) < 1
        or not latents.is_floating_point()
        or type(patch) is not int
        or patch < 1
        or any(size % patch for size in latents.shape[-2:])
    ):
        raise ValueError("Krea2 packing requires nonempty floating BCHW divisible by positive integer patch")
    batch, channels, height, width = latents.shape
    return (
        latents.reshape(batch, channels, height // patch, patch, width // patch, patch)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, height * width // patch**2, channels * patch**2)
    )


def unpack_latents(tokens: Tensor, *, patch: int, height: int, width: int) -> Tensor:
    """Return BCHW from native image tokens, retaining dtype/device and gradients."""
    if (
        tokens.ndim != 3
        or min(tokens.shape) < 1
        or not tokens.is_floating_point()
        or min(patch, height, width) < 1
        or height % patch
        or width % patch
        or tokens.shape[1] != height * width // patch**2
        or tokens.shape[2] % patch**2
    ):
        raise ValueError("Krea2 unpacking requires matching BNC patch geometry")
    batch, _, channels = tokens.shape
    return (
        tokens.reshape(batch, height // patch, width // patch, channels // patch**2, patch, patch)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels // patch**2, height, width)
    )


def image_positions(latents: Tensor, *, patch: int, text_count: int) -> Tensor:
    """Return FP32 B(N+S)3 positions: zero frame, image row/column, then zero text."""
    if (
        latents.ndim != 4
        or min(latents.shape) < 1
        or min(patch, text_count) < 1
        or any(size % patch for size in latents.shape[-2:])
    ):
        raise ValueError("Krea2 positions require BCHW divisible by patch and positive text_count")
    batch, _, height, width = latents.shape
    positions = torch.zeros(height // patch, width // patch, 3, device=latents.device, dtype=torch.float32)
    positions[..., 1] = torch.arange(height // patch, device=latents.device)[:, None]
    positions[..., 2] = torch.arange(width // patch, device=latents.device)[None]
    return F.pad(positions.reshape(1, -1, 3).repeat(batch, 1, 1), (0, 0, 0, text_count))
