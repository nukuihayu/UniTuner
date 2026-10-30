"""Combine restoration terms without owning feature extractors or optimizer phases."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from unituner.config.restoration import RestorationLossWeights
from unituner.training.losses.adversarial import gan_loss
from unituner.training.losses.histogram import histogram_loss
from unituner.training.losses.ms_ssim import ms_ssim_loss
from unituner.training.losses.phase import phase_loss


@dataclass(frozen=True, slots=True)
class RestorationLoss:
    """Carry the differentiable total and independently detached FP32 term metrics."""

    total: Tensor
    metrics: dict[str, Tensor]


def generator_loss(
    generated: Tensor,
    target: Tensor,
    low_latent: Tensor,
    noisy_high: Tensor | None,
    weights: RestorationLossWeights,
    *,
    perceptual: Tensor | None = None,
    frequency_distribution: Tensor | None = None,
    discriminator_logits: tuple[Tensor, ...] | None = None,
    histogram_bins: int = 32,
    phase_period: int = 16,
    phase_cycles: int = 8,
    phase_epsilon: float = 1e-3,
) -> RestorationLoss:
    """Sum FP32 restoration terms and optional color/periodic residual constraints.

    RGB generated/target tensors share nonempty BCHW layout/device in [-1,1].
    Target and the HQ normalized latent are detached internally. Optional Dv3D
    and FDL are already computed scalar losses, retaining their image gradients;
    the method owns frozen feature models. GAN logits must come from a forward
    with discriminator parameters frozen, preserving image gradients and mutable
    spectral-normalization buffers. Missing enabled terms fail explicitly.
    """
    if (
        generated.ndim != 4
        or generated.shape[1] != 3
        or generated.shape != target.shape
        or not generated.numel()
        or generated.device != target.device
        or not generated.is_floating_point()
        or not target.is_floating_point()
    ):
        raise ValueError("Restoration requires matching nonempty floating RGB BCHW images/device")
    for name, weight, term in (("Dv3D", weights.dv3d, perceptual), ("FDL", weights.fdl, frequency_distribution)):
        if weight and (term is None or term.ndim != 0 or term.device != generated.device):
            raise ValueError(f"{name} requires its scalar feature loss on the image device")
    if weights.gan and discriminator_logits is None:
        raise ValueError("GAN requires generator-phase discriminator logits")
    if weights.lrr and (
        noisy_high is None or low_latent.shape != noisy_high.shape or low_latent.device != noisy_high.device
    ):
        raise ValueError("LRR requires matching LQ and frozen HQ target latents")
    with torch.autocast(generated.device.type, enabled=False):
        zero = generated.new_zeros((), dtype=torch.float32)
        terms = {
            "l1": F.l1_loss(generated.float(), target.detach().float()) if weights.l1 else zero,
            "lrr": F.mse_loss(low_latent.float(), noisy_high.detach().float())
            if weights.lrr and noisy_high is not None
            else zero,
            "dv3d": perceptual.float() if weights.dv3d and perceptual is not None else zero,
            "fdl": frequency_distribution.float() if weights.fdl and frequency_distribution is not None else zero,
            "gan": gan_loss(discriminator_logits, True) if weights.gan and discriminator_logits is not None else zero,
            "histogram": histogram_loss(generated, target, histogram_bins) if weights.histogram else zero,
        }
        coefficients = {
            "l1": weights.l1,
            "lrr": weights.lrr,
            "dv3d": weights.dv3d,
            "fdl": weights.fdl,
            "gan": weights.gan,
            "histogram": weights.histogram,
        }
        total = sum((coefficients[name] * value for name, value in terms.items()), start=zero)
        # Keep the primary loss graph/order intact before adding this auxiliary term.
        phase = zero
        if weights.phase:
            phase = phase_loss(generated, target, period=phase_period, cycles=phase_cycles, epsilon=phase_epsilon)
            total = total + weights.phase * phase
        terms["phase"] = phase
        ms_ssim = ms_ssim_loss(generated, target) if weights.ms_ssim else zero
        if weights.ms_ssim:
            total = total + weights.ms_ssim * ms_ssim
        terms["ms_ssim"] = ms_ssim
    return RestorationLoss(total, {name: value.detach() for name, value in terms.items()})
