"""Build model-specific training noise grids independently of inference schedules."""

import math

import torch
from torch import Tensor


def training_sigma_grid(family: str, sampling: str, *, device: torch.device) -> Tensor | None:
    """Return 1000 descending FP32 sigmas; None selects continuous uniform sampling."""
    if sampling == "uniform":
        return None
    defaults = {
        "flux": 3.0,
        "flux2": math.exp(0.8),
        "flux3": 6.93,
        "wan": 5.0,
        "z-image": 3.0,
        "qwen-image": math.exp(0.8),
        "qwen-image-2-1": math.exp(0.8),
        "moss-image": 1.0,
        "krea2": math.exp(0.8),
    }
    if sampling != "shifted":
        raise ValueError("Training timestep sampling must be shifted or uniform")
    factor = defaults[family]
    if family == "flux":
        values = torch.linspace(1.0, 0.003 / 1.002, 1000, dtype=torch.float32, device="cpu")
    elif family == "moss-image":
        values = torch.linspace(1.0, factor / (1000 + factor - 1), 1000, dtype=torch.float32, device="cpu")
    else:
        values = torch.linspace(1.0, 0.0, 1001, dtype=torch.float32, device="cpu")[:-1]
    if family in ("flux2", "qwen-image", "qwen-image-2-1", "krea2"):
        values = factor / (factor + (1 / values - 1))
    else:
        values = factor * values / (1 + (factor - 1) * values)
    if family in ("qwen-image", "qwen-image-2-1"):
        remainder = 1 - values
        values = 1 - remainder / (remainder[-1] / 0.98)
    return values.to(device)


def sample_sigmas(batch_size: int, *, device: torch.device, grid: Tensor | None) -> Tensor:
    if grid is None:
        return torch.rand(batch_size, device=device)
    return grid[torch.randint(len(grid), (batch_size,), device=device)]
