"""Validate tensor values with bounded temporary storage."""

from collections.abc import Iterator

import torch
from torch import Tensor


def _chunks(value: Tensor, size: int) -> Iterator[Tensor]:
    if value.numel() <= size:
        yield value
    elif value.is_contiguous():
        yield from value.view(-1).split(size)
    else:
        axis = next(index for index, width in enumerate(value.shape) if width > 1)
        rows = max(1, size // (value.numel() // value.shape[axis]))
        for part in value.split(rows, dim=axis):
            yield from _chunks(part, size)


def all_finite(value: Tensor, *, dtype: torch.dtype | None = None, chunk_elements: int = 1024**2) -> bool:
    """Bound finite/range-check temporaries, including noncontiguous and FP8 tensors."""
    if chunk_elements <= 0:
        raise ValueError("Finite-check chunk size must be positive")
    for chunk in _chunks(value, chunk_elements):
        checked = chunk if dtype is None else chunk.to(dtype=dtype)
        # Float8 CPU kernels do not uniformly implement isfinite.
        if checked.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            checked = checked.float()
        if not bool(torch.isfinite(checked).all()):
            return False
    return True
