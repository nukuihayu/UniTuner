"""Construct adapter branches for explicit linear geometry and numerical choices."""

import math

import torch
from torch import nn

from unituner.adapters.attachment import LinearBranch
from unituner.adapters.config import AdapterConfig
from unituner.adapters.loha import LoHa
from unituner.adapters.lokr import LoKr
from unituner.adapters.lora import LoRA
from unituner.models.components.quantized import linear_dtype


def _factor_pair(size: int) -> tuple[int, int]:
    """Choose the closest exact integer Kronecker factorization, including primes."""
    first = math.isqrt(size)
    while size % first:
        first -= 1
    return first, size // first


def build_branch(
    projection: nn.Linear,
    config: AdapterConfig,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> LinearBranch:
    """Construct an eager branch matching the selected native linear projection."""
    device = projection.weight.device if device is None else device
    dtype = linear_dtype(projection) if dtype is None else dtype
    match config.algorithm:
        case "lora":
            return LoRA(
                projection.in_features,
                projection.out_features,
                rank=config.rank,
                alpha=config.alpha,
                device=device,
                dtype=dtype,
            )
        case "loha":
            return LoHa(
                projection.in_features,
                projection.out_features,
                rank=config.rank,
                alpha=config.alpha,
                device=device,
                dtype=dtype,
            )
        case "lokr":
            return LoKr(
                _factor_pair(projection.in_features),
                _factor_pair(projection.out_features),
                rank=config.rank,
                alpha=config.alpha,
                device=device,
                dtype=dtype,
            )
