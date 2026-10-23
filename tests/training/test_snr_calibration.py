"""Check streaming SNR candidate selection against independent per-image scalar arithmetic."""

import math

import pytest
import torch

from unitunner.training.methods.mid_timestep import restoration_candidates, restoration_sigma
from unitunner.training.methods.snr_calibration import SNRCalibration, calibration_rng, estimate_mid_timestep


def test_estimate_matches_analytic_fm_snr_and_restores_posterior_rng():
    """The optimum for unit signal and residual2/3 is sigma0.4, independent of architecture."""
    high = torch.ones(1, 2, 3, dtype=torch.float64)
    low = high + 2 / 3
    for family in ("z-image", "flux", "flux2", "flux3", "moss-image", "qwen-image", "qwen-image-2-1", "krea2"):
        grid = restoration_candidates(family=family, resolution=(1024, 1024)).double()
        expected = (((1 - grid) / grid).square() - 2.25).abs()
        estimate = estimate_mid_timestep(iter([(low, high)]), family=family, resolution=(1024, 1024))
        assert estimate.pairs == 1 and estimate.midtimestep == int(expected.argmin()) + 1
        assert estimate.sigma == grid[expected.argmin()].item()
        assert estimate.mean_snr_error == pytest.approx(expected.min().item(), abs=1e-12)
    before = torch.get_rng_state()
    with calibration_rng(17, torch.device("cpu")):
        first = torch.randn(4)
    with calibration_rng(17, torch.device("cpu")):
        torch.testing.assert_close(torch.randn(4), first, atol=0, rtol=0)
    assert torch.equal(before, torch.get_rng_state())


def test_flux_middle_index_uses_resolution_shift_and_calibration_same_grid():
    """Check Trainer's reversed FLUX schedule against independent FP32 arithmetic."""
    for width, height in ((1024, 1024), (512, 768)):
        tokens = (width // 16) * (height // 16)
        mu = 0.5 + (tokens - 256) * (1.15 - 0.5) / (4096 - 256)
        grid = torch.linspace(1, 0, 1000, dtype=torch.float32)
        expected = math.exp(mu) / (math.exp(mu) + 1 / grid - 1)
        candidates = restoration_candidates(family="flux", resolution=(width, height))
        torch.testing.assert_close(candidates, expected[1:-1].flip(0), atol=1e-7, rtol=1e-6)
        for index in (1, 244, 998):
            sigma = restoration_sigma(index, family="flux", resolution=(width, height))
            assert sigma == candidates[index - 1].item()
            assert sigma == pytest.approx(expected[-(index + 1)].item(), abs=1e-7)
    assert restoration_sigma(244, family="z-image", resolution=(1024, 1024)) == 0.244
    assert restoration_sigma(244, family="flux", resolution=(1024, 1024)) != 0.244


def test_snr_matches_scalar_objective_with_unequal_resolutions_and_first_tie():
    sigmas = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    calibration = SNRCalibration(sigmas)
    images = [([1.0, 2.0], [2.0, 3.0]), ([2.0, 4.0, 6.0], [1.0, 3.0, 5.0])]
    expected = []
    before = torch.get_rng_state()
    for low, high in images:
        calibration.add(torch.tensor([low], dtype=torch.float64), torch.tensor([high], dtype=torch.float64))
    for sigma in sigmas.tolist():
        error = 0.0
        for low, high in images:
            signal = sum(x * x for x in high) / len(high)
            residual = sum((x - y) ** 2 for x, y in zip(low, high, strict=True)) / len(high)
            error += abs(((1 - sigma) / sigma) ** 2 * signal - signal / max(residual, 1e-12))
        expected.append(error)
    torch.testing.assert_close(calibration.error, torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    assert calibration.best_sigma() == sigmas[expected.index(min(expected))].item()
    assert calibration.count == 2
    assert torch.equal(before, torch.get_rng_state())
    tie = SNRCalibration(torch.tensor([0.5, 0.5]))
    tie.add(torch.ones(1, 2), torch.ones(1, 2))
    assert tie.best_sigma() == 0.5
    assert math.isfinite(tie.error[0])


def test_snr_invalid_observations_leave_accumulator_unchanged():
    calibration = SNRCalibration(torch.tensor([0.25, 0.5]))
    with pytest.raises(ValueError, match="at least one"):
        calibration.best_sigma()
    invalid = [
        (torch.empty(0, 2), torch.empty(0, 2)),
        (torch.ones(1, 2), torch.zeros(1, 2)),
        (torch.ones(1, 2), torch.full((1, 2), float("nan"))),
        (torch.ones(1, 2), torch.ones(1, 3)),
    ]
    for low, high in invalid:
        with pytest.raises(ValueError, match="SNR calibration"):
            calibration.add(low, high)
        assert calibration.count == 0
        assert not calibration.error.count_nonzero()
    for candidates in (torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([float("nan")]), torch.tensor([1])):
        with pytest.raises(ValueError, match="SNR candidates"):
            SNRCalibration(candidates)
