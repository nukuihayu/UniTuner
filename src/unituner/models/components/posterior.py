"""Represent diagonal Gaussian moments with caller-owned posterior sampling noise."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class DiagonalPosterior:
    """Carry raw BCHW/BCTHW mean/log-variance without implicit clamping or RNG.

    Each VAE owns its native log-variance policy. Sampling allocates a
    differentiable result using caller-supplied noise.
    """

    mean: Tensor
    log_variance: Tensor

    def sample(self, noise: Tensor) -> Tensor:
        """Return mean + exp(log_variance/2)*noise with identical shape/dtype/device."""
        if noise.shape != self.mean.shape or noise.dtype != self.mean.dtype or noise.device != self.mean.device:
            raise ValueError("Posterior noise must match mean shape, dtype and device")
        return self.mean + torch.exp(0.5 * self.log_variance) * noise
