"""Assemble cached model latents without copying a single-sample batch."""

import torch
from torch import Tensor


def batch_latents(values: list[Tensor]) -> Tensor:
    """Keep single-record geometry contiguous while preserving native batch stacking."""
    if not values:
        raise ValueError("Cached latent batches must be nonempty")
    return values[0].unsqueeze(0) if len(values) == 1 else torch.stack(values)
