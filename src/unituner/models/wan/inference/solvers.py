"""Advance native Wan Euler, DPM++ and UniPC sampling sessions."""

from typing import Literal

import torch
from torch import Tensor

from unituner.models.wan.inference.schedules import build_schedule
from unituner.models.wan.inference.unipc import UniPCState
from unituner.precision import FLOAT_DTYPES


class WanSolver:
    """Own one floating inference trajectory and bounded solver prediction history.

    The caller supplies finite model velocities and sample tensors on CPU/CUDA.
    Each step allocates a sample, preserves both inputs and consumes no RNG.
    DPM++ uses data prediction sample-sigma*velocity, second-order midpoint,
    first-order startup and a first-order terminal zero-sigma step. UniPC uses
    native order2/bh2 data prediction, correction and final order reduction.
    Nondefault solver policies are unsupported. No model is owned.
    """

    def __init__(self, steps: int, *, solver: Literal["euler", "dpm++", "unipc"], shift: float) -> None:
        if solver not in ("euler", "dpm++", "unipc"):
            raise ValueError("WanSolver supports euler/dpm++/unipc")
        self.schedule = build_schedule(steps, solver=solver, shift=shift)
        self._unipc = UniPCState(self.schedule.sigmas) if solver == "unipc" else None
        self.solver = solver
        self.step_index = 0
        self._previous_data: Tensor | None = None

    @torch.no_grad()
    def step(self, velocity: Tensor, sample: Tensor) -> Tensor:
        """Advance one full trajectory interval for matching floating BCFHW inputs.

        Reject exhausted sessions and history shape/device changes before state
        mutation. Model timesteps are supplied separately from the host schedule;
        ordered steps permit repeated truncated integer model times without an
        ambiguous timestep-to-index search. Returned storage is caller-owned.
        """
        if (
            sample.ndim != 5
            or min(sample.shape) < 1
            or sample.dtype not in FLOAT_DTYPES.values()
            or velocity.shape != sample.shape
            or velocity.dtype != sample.dtype
            or velocity.device != sample.device
            or sample.device.type not in ("cpu", "cuda")
        ):
            raise ValueError("Wan solver requires matching floating BCFHW sample/velocity on CPU/CUDA")
        previous = self._previous_data
        if previous is not None and (
            previous.shape != sample.shape or previous.device != sample.device or previous.dtype != sample.dtype
        ):
            raise ValueError("Wan solver history differs from current sample shape/device")
        index = self.step_index
        if index >= len(self.schedule.timesteps):
            raise ValueError("Wan solver trajectory is exhausted")
        sigmas = self.schedule.sigmas
        sigma_t, sigma_s = sigmas[index + 1], sigmas[index]
        if self.solver == "euler":
            result = sample + velocity * (sigma_t - sigma_s)
            data = None
        elif self._unipc is not None:
            data = sample - sigma_s * velocity
            result = self._unipc.step(data, sample, index)
        else:
            data = sample - sigma_s * velocity
            alpha_t, alpha_s = 1 - sigma_t, 1 - sigma_s
            lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
            lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
            h = lambda_t - lambda_s
            if previous is None or index == len(self.schedule.timesteps) - 1:
                result = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * data
            else:
                sigma_previous = sigmas[index - 1]
                lambda_previous = torch.log(1 - sigma_previous) - torch.log(sigma_previous)
                h_previous = lambda_s - lambda_previous
                ratio = h_previous / h
                derivative = (1.0 / ratio) * (data - previous)
                result = (
                    (sigma_t / sigma_s) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * data
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * derivative
                )
        self._previous_data = data
        self.step_index += 1
        return result.to(sample.dtype)
