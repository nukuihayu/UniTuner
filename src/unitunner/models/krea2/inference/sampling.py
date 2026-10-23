# Derived from Krea2 and pinned Musubi Tuner; see THIRD_PARTY_NOTICES.md.
"""Integrate Krea2 native normalized-time velocities with resolution-aware Euler CFG."""

import math
from itertools import pairwise

import torch
from torch import Tensor

from unitunner.config.sampling import SamplingConfig, validate_image_sampling
from unitunner.models.flux.conditioning import FluxCondition
from unitunner.models.krea2.conditioning import Krea2TextFeatures, image_positions, pack_latents, unpack_latents
from unitunner.models.krea2.transformer import Krea2Transformer


def build_schedule(image_count: int, steps: int, *, mu: float | None = None) -> list[float]:
    """Return native CPU FP32 shifted1→0 times with factor8/patch2 resolution anchors.

    Interpolate log-shift between256² and1280² pixels, with endpoints0.5/1.15.
    Explicit mu selects a constant log-shift for either RAW or Turbo. Preserve
    native FP32 linspace and arithmetic order, including the zero endpoint.
    """
    if type(image_count) is not int or image_count < 1 or type(steps) is not int or steps < 1:
        raise ValueError("Krea2 schedule requires positive integer image count and steps")
    if mu is None:
        slope = (1.15 - 0.5) / (80**2 - 16**2)
        mu = slope * image_count + (0.5 - slope * 16**2)
    if not math.isfinite(mu) or not -80 <= mu <= 80:
        raise ValueError("Krea2 log-shift must be finite in[-80,80] for FP32 arithmetic")
    times = torch.linspace(1, 0, steps + 1, dtype=torch.float32, device="cpu")
    return (math.exp(mu) / (math.exp(mu) + (1.0 / times - 1.0) ** 1.0)).tolist()


@torch.no_grad()
def sample_latents(
    model: Krea2Transformer,
    noise: Tensor,
    positive: Krea2TextFeatures,
    *,
    negative: Krea2TextFeatures | None = None,
    steps: int = 28,
    mu: float | None = None,
    guidance: float = 5.5,
) -> Tensor:
    """Return FP32 BCHW from explicit noise and BSLD captions without ownership changes.

    Native image-first patch order uses channel/row/column. Time is normalized
    directly, without the training entrance's1000 roundtrip. CFG above1 requires
    a negative branch; each branch keeps independent valid-prefix lengths.
    Arithmetic uses Python FP32 schedule scalars in native Euler order. Preserve
    caller mode, RNG, residency and storage; do not decode or mutate parameters.
    """
    parameter = next(model.parameters())
    if (
        noise.ndim != 4
        or min(noise.shape) < 1
        or noise.shape[1] != model.config.in_channels
        or any(size % model.config.patch_size for size in noise.shape[-2:])
        or noise.dtype != torch.float32
        or parameter.dtype != torch.float32
        or parameter.device != noise.device
        or not torch.isfinite(noise).all()
        or not math.isfinite(guidance)
    ):
        raise ValueError("Krea2 sampling requires finite FP32 BCHW noise on the matching model and finite CFG")
    if guidance > 1 and negative is None:
        raise ValueError("Krea2 CFG above1 requires negative text features")
    for branch in (positive, negative) if guidance > 1 else (positive,):
        if branch is None:
            raise ValueError("Krea2 CFG requires negative features")
        if (
            branch.features.ndim != 4
            or branch.features.shape[0] != noise.shape[0]
            or branch.features.shape[2:] != (model.config.text_layers, model.config.text_dim)
            or branch.features.dtype != torch.float32
            or len(branch.lengths) != noise.shape[0]
            or any(type(length) is not int or not 1 <= length <= branch.features.shape[1] for length in branch.lengths)
            or not torch.isfinite(branch.features).all()
        ):
            raise ValueError("Krea2 requires finite FP32 BSLD features and one nonempty valid prefix per image")
    state = pack_latents(noise, patch=model.config.patch_size)
    schedule = build_schedule(state.shape[1], steps, mu=mu)
    positions = image_positions(noise, patch=model.config.patch_size, text_count=positive.features.shape[1])
    context = positive.features.to(noise.device)
    negative_context = negative.features.to(noise.device) if guidance > 1 and negative is not None else None
    negative_positions = (
        image_positions(noise, patch=model.config.patch_size, text_count=negative.features.shape[1])
        if guidance > 1 and negative is not None
        else None
    )
    with torch.autocast(noise.device.type, enabled=False):
        for current, following in pairwise(schedule):
            time = torch.full((noise.shape[0],), current, dtype=noise.dtype, device=noise.device)
            velocity = model(state, context, time, positions, text_lengths=positive.lengths)
            if negative_context is not None and negative_positions is not None and negative is not None:
                unconditional = model(state, negative_context, time, negative_positions, text_lengths=negative.lengths)
                velocity = unconditional + guidance * (velocity - unconditional)
            state = state + (following - current) * velocity
    return unpack_latents(state, patch=model.config.patch_size, height=noise.shape[-2], width=noise.shape[-1])


@torch.no_grad()
def generate_latents(
    model: Krea2Transformer,
    positive: Krea2TextFeatures,
    sampling: SamplingConfig,
    *,
    negative: Krea2TextFeatures | None = None,
    vae_factor: int | None = None,
    condition: FluxCondition | None = None,
    references: tuple[Tensor, ...] = (),
    inpainting_mask: Tensor | None = None,
) -> Tensor:
    """Initialize native Krea2 noise and sample without changing model mode.

    Args:
        model: Borrowed native transformer; the caller owns eval mode and residency.
        positive: Native selected-layer BLD features with host prefix lengths.
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
    validate_image_sampling(sampling, family="krea2", variant=model.config.variant)
    if not isinstance(positive, Krea2TextFeatures) or (
        negative is not None and (not isinstance(negative, Krea2TextFeatures))
    ):
        raise ValueError("Krea2 requires selected-layer text features")
    if condition is not None or references or inpainting_mask is not None or (vae_factor not in (None, 8)):
        raise ValueError("Krea2 generation requires native factor8 vae without image conditions")
    if model.config.patch_size != 2 or model.config.in_channels != 16 or positive.features.shape[0] != 1:
        raise ValueError("Krea2 generation requires native patch2/latent16 geometry and one caption")
    positive = Krea2TextFeatures(positive.features.float(), positive.lengths)
    if negative is not None:
        negative = Krea2TextFeatures(negative.features.float(), negative.lengths)
    parameter = next(model.parameters())
    noise = torch.randn(
        (1, 16, sampling.height // 8, sampling.width // 8),
        generator=torch.Generator(device=parameter.device).manual_seed(sampling.seed),
        device=parameter.device,
        dtype=torch.float32,
    )
    return sample_latents(
        model,
        noise,
        positive,
        negative=negative,
        steps=28 if sampling.steps is None else sampling.steps,
        mu=sampling.shift,
        guidance=5.5 if sampling.guidance is None else sampling.guidance,
    )
