"""Build native Wan sampling grids independently of training noise fractions."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class WanSchedule:
    """Own read-only CPU FP32 sigmas and CPU native model times.

    Sigmas contain the terminal zero; timesteps contain one entry per prediction.
    DPM++/UniPC times truncate a shifted FP64 grid to int64 before the sigmas
    round to FP32. Euler times stay FP32. Never recover sigma from these times.
    """

    sigmas: Tensor
    timesteps: Tensor

    def high_noise_steps(self, *, boundary: float) -> tuple[bool, ...]:
        """Plan expert selection with native FP32 time/1000 >= normalized boundary.

        Match the reference's inclusive boundary after model-time truncation,
        rather than comparing the unrounded sigma. Return host decisions so the
        sampling loop need not synchronize device scalar booleans each step.
        """
        if not math.isfinite(boundary) or not 0 <= boundary <= 1:
            raise ValueError("Wan expert boundary must be finite and in[0,1]")
        return tuple((self.timesteps / 1000.0 >= boundary).tolist())


def build_schedule(steps: int, *, solver: Literal["euler", "dpm++", "unipc"], shift: float) -> WanSchedule:
    """Build the exact static native grid with 1000 training timesteps.

    DPM++ starts at one and applies one FP64 rational shift. UniPC starts at the
    FP32-rounded999/1000 constructor endpoint, interpolates in FP64, then shifts.
    Euler uses a CPU FP32 linspace and rational shift. Preserve these distinct
    arithmetic paths; neither the direct training time offset1 nor a second
    sampling shift applies here. Allocate fresh host tensors without RNG draws.
    """
    if type(steps) is not int or steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("Wan sampling requires positive integer steps and finite positive shift")
    if solver == "euler":
        sigmas = torch.linspace(1, 0, steps + 1, dtype=torch.float32, device="cpu")
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        times = sigmas[:-1] * 1000
    elif solver in ("dpm++", "unipc"):
        start = 1.0 if solver == "dpm++" else float(np.float32(0.999))
        grid = np.linspace(start, 0, steps + 1)[:-1]
        grid = shift * grid / (1 + (shift - 1) * grid)
        times = torch.from_numpy(grid * 1000).to(torch.int64)
        sigmas = torch.from_numpy(np.concatenate([grid, [0]]).astype(np.float32))
    else:
        raise ValueError("Wan sampling solver must be euler/dpm++/unipc")
    if not torch.isfinite(sigmas).all() or not torch.all(sigmas[:-1] > sigmas[1:]) or sigmas[0] > 1:
        raise ValueError("Wan sampling sigmas must be finite and strictly decreasing within[0,1]")
    return WanSchedule(sigmas, times)
