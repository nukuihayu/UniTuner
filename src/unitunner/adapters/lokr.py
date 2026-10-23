"""Compute native Kronecker linear deltas with explicit factor geometry."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unitunner.adapters.ops.lokr import scaled_kronecker


class LoKr(nn.Module):
    """Apply kron(left, right_up @ right_down) * alpha/rank.

    Left has [out_left, in_left]; the right factor has [out_right, in_right].
    Full-right mode retains the old unit scaling rule and does not claim that
    its declared rank controls matrix rank. Gradients include inputs and all
    active factors. Shape choices are explicit rather than inferred from names.
    """

    alpha: Tensor

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_shape: tuple[int, int],
        *,
        rank: int,
        alpha: float,
        full_right: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features: int = in_shape[0] * in_shape[1]
        self.out_features: int = out_shape[0] * out_shape[1]
        if min(*in_shape, *out_shape, rank) <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoKr dimensions, rank and finite alpha must be positive")
        if not dtype.is_floating_point:
            raise ValueError("LoKr requires a floating dtype")
        self.left: nn.Parameter = nn.Parameter(torch.empty(out_shape[0], in_shape[0], device=device, dtype=dtype))
        self.right: nn.Parameter | None = None
        self.right_up: nn.Parameter | None = None
        self.right_down: nn.Parameter | None = None
        self.rank: int = rank
        self.register_buffer("alpha", torch.tensor(float(rank) if full_right else alpha, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.left, a=math.sqrt(5))
        if full_right:
            self.right = nn.Parameter(torch.zeros(out_shape[1], in_shape[1], device=device, dtype=dtype))
        else:
            self.right_up = nn.Parameter(torch.empty(out_shape[1], rank, device=device, dtype=dtype))
            self.right_down = nn.Parameter(torch.zeros(rank, in_shape[1], device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.right_up, a=math.sqrt(5))

    @property
    def scale(self) -> Tensor:
        """Derive live scaling from the serialized alpha instead of a stale Python copy."""
        return self.alpha / self.rank

    def _right_factor(self) -> Tensor:
        """Materialize the small right factor in the active autocast dtype."""
        if self.right is not None:
            right = self.right
        else:
            if self.right_up is None or self.right_down is None:
                raise RuntimeError("LoKr factors are incomplete")
            right = self.right_up @ self.right_down
        return right

    def delta(self) -> Tensor:
        """Return [out_left*out_right, in_left*in_right] with exact autograd."""
        return torch.kron(self.left.contiguous(), self._right_factor().contiguous()) * self.scale

    # Torch declares forward as a mutable Callable[..., Any], although subclasses narrow it.
    def forward(self, inputs: Tensor) -> Tensor:  # ty: ignore[invalid-mutable-override]
        """Return differentiable [..., out_left*out_right] additive predictions."""
        right = self._right_factor()
        dtype = (
            torch.get_autocast_dtype(inputs.device.type)
            if torch.is_autocast_enabled(inputs.device.type)
            else torch.promote_types(self.left.dtype, right.dtype)
        )
        return F.linear(inputs, scaled_kronecker(self.left, right, self.scale, dtype))
