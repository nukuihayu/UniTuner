"""Accumulate the original per-image SNR matching objective over explicit sigma candidates."""

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor

from unitunner.training.methods.mid_timestep import restoration_candidates


@contextmanager
def calibration_rng(seed: int, device: torch.device) -> Generator[None]:
    """Isolate calibration posterior draws from training and other GPU RNG streams."""
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield


@dataclass(frozen=True, slots=True)
class MidTimestepEstimate:
    """Describe the selected native middle index, exact sigma and mean SNR error."""

    midtimestep: int
    sigma: float
    mean_snr_error: float
    pairs: int


def estimate_mid_timestep(
    pairs: Iterable[tuple[Tensor, Tensor]], *, family: str, resolution: tuple[int, int]
) -> MidTimestepEstimate:
    """Estimate OMGSR FM time from a streaming sequence of normalized LQ/HQ latents.

    Each pair is matching floating batch-first tensors in the VAE's native
    normalized units. Use paper equations18/19: mean per-image absolute SNR
    error across the native candidate grid, with each image equally weighted.
    Only one pair and O(candidate_count) moments are retained; model/RNG and
    input gradients are untouched. Return the index and actual sigma separately
    so FLUX's shifted index is never confused with normalized model time.
    """
    calibration = SNRCalibration(restoration_candidates(family=family, resolution=resolution))
    for low, high in pairs:
        calibration.add(low, high)
    sigma = calibration.best_sigma()
    index = int(calibration.error.argmin())
    return MidTimestepEstimate(index + 1, sigma, float(calibration.error[index]) / calibration.count, calibration.count)


class SNRCalibration:
    """Own CPU FP64 candidate errors with constant storage independent of image count.

    Match observed SNR E[high²]/max(E[(low-high)²],1e-12) against
    ((1-sigma)/sigma)² E[high²]. Each image has equal weight irrespective of
    resolution; this preserves the original signal-dependent candidate objective.
    Candidates are ordered; ties select the first. No RNG or gradients are used.
    """

    def __init__(self, sigmas: Tensor) -> None:
        if (
            sigmas.ndim != 1
            or not sigmas.numel()
            or not sigmas.is_floating_point()
            or not torch.isfinite(sigmas).all()
            or not ((sigmas > 0) & (sigmas < 1)).all()
        ):
            raise ValueError("SNR candidates must be a finite floating vector strictly inside (0,1)")
        self.sigmas = sigmas.detach().to(device="cpu", dtype=torch.float64).clone()
        self.error = torch.zeros_like(self.sigmas)
        self.count = 0

    def add(self, low: Tensor, high: Tensor) -> None:
        """Accumulate detached image-wise errors without retaining inputs or changing RNG.

        Inputs share nonempty batch-first shape/device and floating dtype, in
        model-normalized latent units. Compute powers in FP64 on their device,
        then transfer per-image moments to CPU. Invalid/overflowing observations
        raise before any accumulator mutation.
        """
        if (
            low.shape != high.shape
            or low.ndim < 2
            or not low.numel()
            or low.device != high.device
            or low.dtype != high.dtype
            or not low.is_floating_point()
        ):
            raise ValueError("SNR calibration requires nonempty matching floating latent pairs")
        low, high = low.detach().double().flatten(1), high.detach().double().flatten(1)
        signal = high.square().mean(1).cpu()
        residual = (low - high).square().mean(1).cpu()
        if not torch.isfinite(signal).all() or not torch.isfinite(residual).all() or (signal <= 0).any():
            raise ValueError("SNR calibration requires finite latents and nonzero signal")
        observed = signal / residual.clamp_min(1e-12)
        expected = ((1 - self.sigmas) / self.sigmas).square()[:, None] * signal[None]
        error = self.error + (expected - observed[None]).abs().sum(1)
        if not torch.isfinite(error).all():
            raise ValueError("SNR calibration error exceeds FP64 range")
        self.error.copy_(error)
        self.count += low.shape[0]

    def best_sigma(self) -> float:
        """Return the first least-error candidate, rejecting an empty observation stream."""
        if not self.count:
            raise ValueError("SNR calibration requires at least one pair")
        return float(self.sigmas[self.error.argmin()])
