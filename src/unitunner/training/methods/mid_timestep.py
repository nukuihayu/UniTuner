"""Evaluate one-step restoration flow without adding noise to the LQ input."""

import torch
from torch import Tensor

from unitunner.models.flux.inference.sampling import build_schedule


def _validate_time_grid(family: str, resolution: tuple[int, int]) -> None:
    if family not in ("z-image", "flux", "flux2", "flux3", "moss-image", "qwen-image", "qwen-image-2-1", "krea2"):
        raise ValueError("A new restoration family must declare its native time grid")
    if any(type(size) is not int or size < 16 or size % 16 for size in resolution):
        raise ValueError("Restoration time grids require positive 16-aligned pixel dimensions")


def restoration_sigma(midtimestep: int, *, family: str, resolution: tuple[int, int]) -> float:
    """Resolve a restoration index to a dimensionless noise fraction, without RNG or IO.

    T is an integer in 1..998. FLUX.1 follows Trainer's reversed 999-step
    resolution-shifted schedule, with packed token count (H//16)*(W//16).
    Other families use T/1000. Resolution is the configured pixel (W,H)
    crop, not a latent shape. Native model time/sign conversion happens later.
    """
    if type(midtimestep) is not int or not 1 <= midtimestep <= 998:
        raise ValueError("Restoration midtimestep must be an integer in 1..998")
    _validate_time_grid(family, resolution)
    if family == "flux":
        width, height = resolution
        return build_schedule(999, (width // 16) * (height // 16))[-(midtimestep + 1)]
    return midtimestep / 1000


def restoration_candidates(*, family: str, resolution: tuple[int, int]) -> Tensor:
    """Return ascending CPU FP32 sigma candidates for indices 1..998.

    FLUX.1 uses its resolution-shifted grid, matching fixed-time training.
    Other flow families use t/1000. Calibration retains equal per-image SNR
    weighting; this function constructs no random or model state.
    """
    _validate_time_grid(family, resolution)
    if family == "flux":
        width, height = resolution
        return torch.tensor(build_schedule(999, (width // 16) * (height // 16))[1:-1][::-1], dtype=torch.float32)
    return torch.arange(1, 999, dtype=torch.float32, device="cpu") / 1000


def _validate_pair(left: Tensor, right: Tensor, sigma: float) -> None:
    if (
        not 0 < sigma < 1
        or left.shape != right.shape
        or left.ndim < 2
        or not left.numel()
        or left.device != right.device
        or not left.is_floating_point()
        or not right.is_floating_point()
    ):
        raise ValueError("Restoration requires matching nonempty floating batched tensors/device and 0 < sigma < 1")


def flow_target(clean: Tensor, noise: Tensor, sigma: float) -> Tensor:
    """Return (1-sigma)*clean + sigma*noise in the original input dtype/layout.

    Inputs use normalized model latent units and identical floating dtype/device;
    caller supplies unit Gaussian noise. Allocate a new tensor with gradients to
    both inputs. Sigma is a finite fraction strictly inside (0,1).
    """
    _validate_pair(clean, noise, sigma)
    if clean.dtype != noise.dtype:
        raise ValueError("Restoration clean/noise dtypes must match")
    return (1 - sigma) * clean + sigma * noise


def one_step(low: Tensor, prediction: Tensor, sigma: float, *, flow_sign: int) -> Tensor:
    """Return low - sigma*flow_sign*prediction using FP32 Euler arithmetic.

    Inputs share batch-first layout/device; mixed floating dtypes are allowed.
    Use sign +1 for noise-minus-data, -1 for Z-Image data-minus-noise.
    Preserve gradients to both inputs without changing either tensor or global RNG.
    """
    _validate_pair(low, prediction, sigma)
    if flow_sign not in (-1, 1):
        raise ValueError("Restoration velocity sign must be +1 or -1")
    return low.float() - sigma * flow_sign * prediction.float()
