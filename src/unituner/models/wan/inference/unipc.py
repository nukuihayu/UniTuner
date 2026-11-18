"""Compute the native second-order data-prediction UniPC bh2 predictor/corrector."""

import torch
from torch import Tensor


class UniPCState:
    """Own at most two data predictions and one corrected sample for one trajectory.

    The enclosing solver validates floating BCFHW geometry and ordered step indices.
    Sigma coefficients stay on CPU as in the native scheduler; the order-two
    corrector's small linear system executes on the sample device. Only the
    default predict-x0/bh2/order2/lower-order-final policy is implemented.
    """

    def __init__(self, sigmas: Tensor) -> None:
        self.sigmas = sigmas
        self.history: tuple[Tensor, ...] = ()
        self.last_sample: Tensor | None = None
        self.last_order = 1

    def step(self, data: Tensor, sample: Tensor, index: int) -> Tensor:
        """Correct the incoming sample, then predict the next one transactionally.

        The current data prediction is computed from the incoming uncorrected
        sample by the caller and is never recomputed after correction. Persist
        the corrected sample and current data only after the predictor succeeds.
        """
        if self.last_sample is not None:
            sample = self._correct(data, sample, index)
        history = (*self.history[-1:], data)
        order = min(2, len(self.sigmas) - 1 - index, index + 1)
        result = self._predict(sample, history, index, order)
        self.history, self.last_sample, self.last_order = history, sample.clone(), order
        return result

    def _predict(self, sample: Tensor, history: tuple[Tensor, ...], index: int, order: int) -> Tensor:
        """Apply the native order-one/two bh2 predictor with terminal order reduction."""
        sigma_t, sigma_s = self.sigmas[index + 1], self.sigmas[index]
        alpha_t, alpha_s = 1 - sigma_t, 1 - sigma_s
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s
        phi = torch.expm1(-h)
        base = sigma_t / sigma_s * sample - alpha_t * phi * history[-1]
        if order == 1:
            residual = 0
        else:
            sigma_previous = self.sigmas[index - 1]
            lambda_previous = torch.log(1 - sigma_previous) - torch.log(sigma_previous)
            ratio = (lambda_previous - lambda_s) / h
            derivatives = torch.stack([(history[-2] - history[-1]) / ratio], dim=1)
            coefficients = torch.tensor([0.5], dtype=sample.dtype, device=sample.device)
            residual = torch.einsum("k,bkc...->bc...", coefficients, derivatives)
        return base - alpha_t * phi * residual

    def _correct(self, data: Tensor, sample: Tensor, index: int) -> Tensor:
        """Apply the bh2 corrector using the preceding predictor's effective order."""
        if self.last_sample is None:
            raise RuntimeError("UniPC correction requires an owned previous sample")
        sigma_t, sigma_s = self.sigmas[index], self.sigmas[index - 1]
        alpha_t, alpha_s = 1 - sigma_t, 1 - sigma_s
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s
        hh = -h
        phi = torch.expm1(hh)
        m0 = self.history[-1]
        base = sigma_t / sigma_s * self.last_sample - alpha_t * phi * m0
        if self.last_order == 1:
            coefficients = torch.tensor([0.5], dtype=sample.dtype, device=sample.device)
            residual = 0
        else:
            sigma_previous = self.sigmas[index - 2]
            lambda_previous = torch.log(1 - sigma_previous) - torch.log(sigma_previous)
            ratio = (lambda_previous - lambda_s) / h
            ratios = torch.tensor([ratio, 1.0], dtype=sample.dtype, device=sample.device)
            matrix = torch.stack([torch.pow(ratios, power) for power in (0, 1)])
            phi_next = phi / hh - 1
            rhs = torch.tensor(
                [phi_next / phi, (phi_next / hh - 0.5) * 2 / phi], dtype=sample.dtype, device=sample.device
            )
            coefficients = torch.linalg.solve(matrix.float(), rhs.float()).to(sample.dtype)
            derivatives = torch.stack([(self.history[-2] - m0) / ratio], dim=1)
            residual = torch.einsum("k,bkc...->bc...", coefficients[:-1], derivatives)
        return base - alpha_t * phi * (residual + coefficients[-1] * (data - m0))
