"""Compute low-rank linear deltas without owning attachment, merge or file IO."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRA(nn.Module):
    """Compute alpha/rank * (x @ down.T) @ up.T on [..., in_features].

    Factors use the requested floating dtype/device. The up factor starts at
    zero. Forward preserves the factor dtype under normal Torch/autocast rules
    and differentiates inputs and both factors. No base projection is retained.
    """

    alpha: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        if min(in_features, out_features, rank) <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA dimensions, rank and finite alpha must be positive")
        if not dtype.is_floating_point:
            raise ValueError("LoRA requires a floating dtype")
        self.down: nn.Parameter = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.up: nn.Parameter = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        self.rank: int = rank
        self.register_buffer("alpha", torch.tensor(alpha, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    # Torch declares forward as a mutable Callable[..., Any], although subclasses narrow it.
    def forward(self, inputs: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        """Return a differentiable [..., out_features] additive prediction."""
        return F.linear(F.linear(inputs, self.down), self.up) * self.scale

    @property
    def scale(self) -> Tensor:
        """Derive live scaling from the serialized alpha instead of a stale Python copy."""
        return self.alpha / self.rank

    def delta(self) -> Tensor:
        """Return differentiable [out_features, in_features] weight delta."""
        return (self.up @ self.down) * self.scale
