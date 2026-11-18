"""Check one-expert microbatches retain the native marginal sigma distribution."""

import torch

from unituner.training.timesteps import training_sigma_grid
from unituner.training.workflows.wan.run import _sample_expert_batch


def test_grid_batch_routing_preserves_domain_weights_and_times():
    device = torch.device("cpu")
    grid = training_sigma_grid("wan", "shifted", device=device)
    boundary = 0.875
    high_mask = grid * 1000 >= boundary * 1000
    high_grid, low_grid = grid[high_mask], grid[~high_mask]
    torch.manual_seed(18)
    batches = [
        _sample_expert_batch(
            8,
            device=device,
            grid=grid,
            boundary=boundary,
            low_grid=low_grid,
            high_grid=high_grid,
        )
        for _ in range(400)
    ]
    for sigmas, high in batches:
        assert bool(torch.all(sigmas * 1000 >= boundary * 1000)) == high
    high_share = sum(high for _, high in batches) / len(batches)
    assert abs(high_share - high_mask.float().mean().item()) < 0.08
    assert abs(torch.cat([values for values, _ in batches]).mean().item() - grid.mean().item()) < 0.05


def test_uniform_batch_routing_uses_the_native_time_offset():
    device = torch.device("cpu")
    torch.manual_seed(27)
    batches = [
        _sample_expert_batch(
            8,
            device=device,
            grid=None,
            boundary=0.5,
            low_grid=None,
            high_grid=None,
        )
        for _ in range(400)
    ]
    for sigmas, high in batches:
        assert bool(torch.all(sigmas * 1000 + 1 >= 500)) == high
    assert abs(sum(high for _, high in batches) / len(batches) - 0.501) < 0.08
    assert abs(torch.cat([values for values, _ in batches]).mean().item() - 0.5) < 0.05
