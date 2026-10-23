"""Integrate Z-Image reverse-flow predictions with the native FP32 Euler solver."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from unitunner.config.sampling import SamplingConfig, validate_image_sampling
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux.conditioning import FluxCondition, FluxTextFeatures
from unitunner.models.z_image.transformer import ZImageTransformer


@dataclass(frozen=True, slots=True)
class ZImageSchedule:
    """Keep separately rounded FP32 noise fractions and native 0-1000 timesteps."""

    sigmas: Tensor
    timesteps: Tensor


def build_schedule(steps: int, *, shift: float = 3.0) -> ZImageSchedule:
    """Return CPU FP32 sigmas [steps+1] and native timesteps [steps].

    Preserve the native float64 linspace(1000,1,steps+1)[:-1]/1000 before
    rational shifting and FP32 rounding; torch.linspace has different rounding.
    """
    if isinstance(steps, bool) or steps < 1 or not math.isfinite(shift) or shift <= 0:
        raise ValueError("Z-Image requires positive steps and finite positive shift")
    sigma = np.linspace(1000, 1, steps + 1)[:-1] / 1000
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return ZImageSchedule(
        torch.cat((torch.from_numpy(sigma).float(), torch.zeros(1))), torch.from_numpy(sigma * 1000).float()
    )


@torch.no_grad()
def sample_latents(
    model: ZImageTransformer,
    noise: Tensor,
    positive: Caption,
    *,
    negative: Caption | None = None,
    steps: int | None = None,
    shift: float = 3.0,
    guidance: float | None = None,
) -> Tensor:
    """Integrate explicit FP32 BCHW noise into model-normalized BCHW latents.

    Model inputs use the model's parameter dtype; the Euler state stays FP32.
    Native time is 1-sigma and solver velocity is minus the native prediction.
    Guidance deliberately preserves positive+guidance*(positive-negative) for
    guidance>1. Missing negative means zero features with the positive mask.
    Omitted steps/guidance use base 50/4 or Turbo 9/0.
    The caller owns model mode/residency; no global RNG or device cleanup occurs.
    """
    if steps is None:
        steps = 9 if model.config.variant == "turbo" else 50
    if guidance is None:
        guidance = 0.0 if model.config.variant == "turbo" else 4.0
    if noise.ndim != 4 or noise.dtype != torch.float32 or noise.shape[1] != model.config.in_channels:
        raise ValueError("Sampling noise requires FP32 BCHW with configured channels")
    if not math.isfinite(guidance):
        raise ValueError("Z-Image guidance must be finite")
    parameter = next(model.parameters())
    if noise.device != parameter.device:
        raise ValueError("Sampling noise and model must share a device")
    schedule = build_schedule(steps, shift=shift)
    sigmas = schedule.sigmas.to(noise.device)
    native_timesteps = schedule.timesteps.to(noise.device)
    positive_features = positive.features.to(device=noise.device, dtype=parameter.dtype)
    positive_mask = None if positive.mask is None else positive.mask.to(noise.device)
    negative_features, negative_mask = positive_features, positive_mask
    if guidance > 1:
        if negative is None:
            negative = Caption(torch.zeros_like(positive_features), positive.mask)
        negative_features = negative.features.to(device=noise.device, dtype=parameter.dtype)
        negative_mask = None if negative.mask is None else negative.mask.to(noise.device)
    latents = noise.clone()
    for index in range(steps):
        timesteps = (1000 - native_timesteps[index].expand(latents.shape[0])) / 1000
        inputs = latents.to(parameter.dtype).unsqueeze(2)
        prediction = model(inputs, timesteps, positive_features, positive_mask)
        if guidance > 1:
            if negative is None:
                raise RuntimeError("Negative conditioning was not constructed")
            unconditional = model(inputs, timesteps, negative_features, negative_mask)
            prediction = prediction + guidance * (prediction - unconditional)
        latents = latents + (sigmas[index + 1] - sigmas[index]) * -prediction.squeeze(2).float()
    return latents


@torch.no_grad()
def generate_latents(
    model: ZImageTransformer,
    positive: Caption,
    sampling: SamplingConfig,
    *,
    negative: Caption | None = None,
    vae_factor: int | None = None,
    condition: FluxCondition | None = None,
    references: tuple[Tensor, ...] = (),
    inpainting_mask: Tensor | None = None,
) -> Tensor:
    """Initialize native Z-Image noise and sample without changing model mode.

    Args:
        model: Borrowed native transformer; the caller owns eval mode and residency.
        positive: Native BLD features and optional BL validity mask.
            Features keep their encoder dtype until the native sampler places them.
        sampling: Pixel width/height, local noise seed, steps, guidance and native
            solver controls. Defaults and validation follow this model family.
        negative: Native unconditional features, or None when guidance does not
            require them. FLUX accepts no negative conditioning.
        vae_factor: Pixel-to-latent spatial factor, or None for the native default.
        condition: Packed FLUX image condition; other families reject it.
        references: Ordered native clean reference means; only reference-capable
            models accept nonempty tuples. Each mean retains its native layout.
        inpainting_mask: Native latent-grid repaint mask for Qwen models, or None;
            other families reject it rather than silently ignoring it.

    Returns:
        Model-native BCHW latents on the model device without a gradient graph.
        Seed-local noise draw order, BF16 rounding where native, solver defaults
        and input tensors remain unchanged; no global RNG is advanced.

    Raises:
        ValueError: Features, conditions or dimensions violate the native contract.
    """
    validate_image_sampling(sampling, family="z-image", variant=model.config.variant, version=None)
    if isinstance(positive, FluxTextFeatures):
        raise ValueError("Flux requires T5/CLIP features; other image models require Caption")
    if condition is not None:
        raise ValueError("Packed image conditions require Flux")
    if references:
        raise ValueError("Ordered image references require FLUX.2 or FLUX3")
    if inpainting_mask is not None:
        raise ValueError("Packed inpainting masks require Qwen Image")
    if vae_factor is None:
        vae_factor = 8
    granularity = vae_factor * 2
    if vae_factor < 1 or sampling.height % granularity or sampling.width % granularity:
        raise ValueError("Sampling dimensions must match vae factor and spatial patch size")
    parameter = next(model.parameters())
    generator = torch.Generator(device="cpu").manual_seed(sampling.seed)
    if not isinstance(positive, Caption):
        raise TypeError("Image model requires Caption")
    noise = torch.randn(
        (1, model.config.in_channels, sampling.height // vae_factor, sampling.width // vae_factor),
        generator=generator,
    ).to(parameter.device)
    return sample_latents(
        model,
        noise,
        positive,
        negative=negative,
        steps=sampling.steps,
        shift=3.0 if sampling.shift is None else sampling.shift,
        guidance=sampling.guidance,
    )
