"""Assemble variable-length native text tensors without loading encoders or training state."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class Caption:
    """Carry BLD features, optional BL validity and optional host prefix lengths.

    batch_captions records lengths before padding, avoiding device transfers for
    model-native ragged layouts. Arbitrary external masks need not be prefixes
    and do not imply these optional lengths.
    """

    features: Tensor
    mask: Tensor | None = None
    lengths: tuple[int, ...] | None = None


def batch_captions(rows: Sequence[Tensor]) -> Caption:
    """Stack nonempty LD captions, padding sequence lengths only for batching.

    All rows share floating dtype/device/width. A single caption has no mask,
    retaining native model padding semantics. Multiple rows get explicit
    validity masks; each transformer owns its internal token padding.
    Returns newly allocated differentiable BLD features, without detaching rows.
    """
    if not rows or any(row.ndim != 2 or min(row.shape) <= 0 or not row.is_floating_point() for row in rows):
        raise ValueError("Captions require nonempty floating LD tensors")
    first = rows[0]
    if any(row.shape[1] != first.shape[1] or row.dtype != first.dtype or row.device != first.device for row in rows):
        raise ValueError("Caption widths, dtypes and devices must agree")
    maximum = max(row.shape[0] for row in rows)
    features = torch.stack([F.pad(row, (0, 0, 0, maximum - row.shape[0])) for row in rows])
    mask = None
    if len(rows) > 1:
        lengths = torch.tensor([row.shape[0] for row in rows], device=first.device)
        mask = torch.arange(maximum, device=first.device)[None] < lengths[:, None]
    return Caption(features, mask, tuple(row.shape[0] for row in rows))
