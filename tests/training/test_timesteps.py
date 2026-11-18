"""Verify model training grids, endpoint policies and resumable sampling."""

import math

import pytest
import torch

from unituner.training.timesteps import sample_sigmas, training_sigma_grid

CPU = torch.device("cpu")
FAMILIES = ("flux", "flux2", "flux3", "wan", "z-image", "qwen-image", "qwen-image-2-1", "moss-image", "krea2")


@pytest.mark.parametrize("family", FAMILIES)
def test_training_grid_matches_model_recipe(family):
    grid = training_sigma_grid(family, "shifted", device=CPU)
    assert grid.shape == (1000,) and grid.dtype == torch.float32
    assert grid[0] == 1 and grid[-1] > 0
    assert torch.all(grid[:-1] > grid[1:])
    factor = {
        "flux": 3,
        "flux2": math.exp(0.8),
        "flux3": 6.93,
        "wan": 5,
        "z-image": 3,
        "qwen-image": math.exp(0.8),
        "qwen-image-2-1": math.exp(0.8),
        "moss-image": 1,
        "krea2": math.exp(0.8),
    }[family]
    for index in (0, 100, 500, 999):
        u = 1 - index / 1000
        if family == "flux":
            u = 1 - index * (1 - 0.003 / 1.002) / 999
        expected = factor * u / (1 + (factor - 1) * u)
        if family.startswith("qwen-image"):
            minimum = factor * 0.001 / (1 + (factor - 1) * 0.001)
            expected = 1 - (1 - expected) * 0.98 / (1 - minimum)
        assert grid[index].item() == pytest.approx(expected, abs=2e-7)


@pytest.mark.parametrize("sampling", ["shifted", "uniform"])
def test_sampling_replays_rng_and_preserves_expected_distribution(sampling):
    grid = training_sigma_grid("qwen-image-2-1", sampling, device=CPU)
    with torch.random.fork_rng():
        torch.manual_seed(38)
        state = torch.get_rng_state()
        values = sample_sigmas(20000, device=CPU, grid=grid)
        torch.set_rng_state(state)
        assert torch.equal(values, sample_sigmas(20000, device=CPU, grid=grid))
    if grid is None:
        assert values.min() >= 0 and values.max() < 1
        assert values.mean().item() == pytest.approx(0.5, abs=0.01)
    else:
        assert torch.isin(values, grid).all()
        assert values.mean().item() == pytest.approx(0.638, abs=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_grid_matches_cpu_and_samples_on_device():
    device = torch.device("cuda")
    for family in FAMILIES:
        grid = training_sigma_grid(family, "shifted", device=device)
        assert torch.equal(grid.cpu(), training_sigma_grid(family, "shifted", device=CPU))
        values = sample_sigmas(16, device=device, grid=grid)
        assert values.device.type == "cuda" and torch.isin(values, grid).all()
