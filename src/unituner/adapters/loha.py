"""Compute native low-rank Hadamard deltas with a rounding-preserving eager product fusion."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unituner.adapters.ops.loha import scaled_product


class LoHa(nn.Module):
    """Apply alpha/rank * ((a1 @ b1) * (a2 @ b2)) to [..., in_features].

    The eager derivative includes inputs and all factors. A2 starts at zero;
    all other factors follow the old native normal initialization. This path
    makes no custom-autograd approximation or dropout claim.
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
            raise ValueError("LoHa dimensions, rank and finite alpha must be positive")
        if not dtype.is_floating_point:
            raise ValueError("LoHa requires a floating dtype")
        self.a1: nn.Parameter = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype))
        self.b1: nn.Parameter = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.a2: nn.Parameter = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        self.b2: nn.Parameter = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.rank: int = rank
        self.register_buffer("alpha", torch.tensor(alpha, device=device, dtype=dtype))
        nn.init.normal_(self.a1, std=0.1)
        nn.init.normal_(self.b1, std=1.0)
        nn.init.normal_(self.b2, std=1.0)

    @property
    def scale(self) -> Tensor:
        """Derive live scaling from the serialized alpha instead of a stale Python copy."""
        return self.alpha / self.rank

    def delta(self) -> Tensor:
        """Return the differentiable [out, in] Hadamard delta on factor storage."""
        return scaled_product(self.a1 @ self.b1, self.a2 @ self.b2, self.scale)

    # Torch declares forward as a mutable Callable[..., Any], although subclasses narrow it.
    def forward(self, inputs: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        """Return [..., out] predictions with exact input and factor gradients."""
        return F.linear(inputs, self.delta())
