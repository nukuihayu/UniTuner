# Copyright 2025 The Qwen-Image Team and The HuggingFace Team.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Integrate native Qwen velocity with terminal-stretched schedules and normalized CFG."""

import math

import numpy as np
import torch
from torch import Tensor

from unitunner.config.sampling import SamplingConfig, validate_image_sampling
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux.conditioning import FluxCondition, FluxTextFeatures
from unitunner.models.qwen_image.conditioning import pack_latents, unpack_latents
from unitunner.models.qwen_image.transformer import QwenImageTransformer
from unitunner.precision import FLOAT_DTYPES, sampling_context, sampling_dtype


def build_schedule(steps: int, token_count: int, *, shift: float | None = None, layered: bool = False) -> Tensor:
    """Return CPU FP32 sigmas with native NumPy shift/stretch and final zero.

    Ordinary image/edit uses mu=0.5+(tokens-256)*0.4/(8192-256).
    Layered uses sqrt(source_reference_tokens/256), independent of output layers.
    An explicit shift selects s*t/(1+(s-1)*t). Stretch the last nonzero sigma
    to0.02 before appending0; at least two steps are required. No clipping of
    resolution extrapolation, dtype/default-device dependence or mutable cursor.
    """
    if type(steps) is not int or steps < 2 or type(token_count) is not int or token_count < 1:
        raise ValueError("Qwen sampling requires at least two steps and positive integer token_count")
    values = np.asarray(np.linspace(1.0, 1 / steps, steps), dtype=np.float32)
    if shift is None:
        slope = (0.9 - 0.5) / (8192 - 256)
        mu = (token_count / 256) ** 0.5 if layered else token_count * slope + (0.5 - slope * 256)
        if not math.isfinite(mu) or mu > math.log(torch.finfo(torch.float32).max):
            raise ValueError("Qwen dynamic shift exceeds finite FP32 range")
        factor = math.exp(mu)
        values = factor / (factor + (1 / values - 1) ** 1.0)
    else:
        if not math.isfinite(shift) or shift <= 0 or shift > torch.finfo(torch.float32).max:
            raise ValueError("Qwen shift must be finite, positive and FP32-representable")
        values = shift * values / (1 + (shift - 1) * values)
    remainder = 1 - values
    if remainder[-1] <= 0:
        raise ValueError("Qwen terminal stretching requires a noncollapsed schedule")
    values = 1 - remainder / (remainder[-1] / (1 - 0.02))
    schedule = torch.cat((torch.from_numpy(values).float(), torch.zeros(1, device="cpu", dtype=torch.float32)))
    if not torch.isfinite(schedule).all() or schedule[0] != 1 or not (schedule[:-1] > schedule[1:]).all():
        raise ValueError("Qwen sigmas must strictly decrease from one to zero")
    return schedule


def normalized_guidance(positive: Tensor, negative: Tensor, scale: float) -> Tensor:
    """Apply per-token native CFG norm rescaling; zero guided vectors remain zero.

    Inputs are matching floating BNC patches. Norms reduce only the channel
    axis, and both the guidance expression and normalization retain gradients.
    """
    if positive.shape != negative.shape or positive.dtype != negative.dtype or positive.device != negative.device:
        raise ValueError("Qwen CFG predictions must share shape, dtype and device")
    if not math.isfinite(scale):
        raise ValueError("Qwen CFG scale must be finite")
    combined = negative + scale * (positive - negative)
    norm = torch.norm(combined, dim=-1, keepdim=True)
    denominator = torch.where(norm > 0, norm, torch.ones_like(norm))
    return combined * (torch.norm(positive, dim=-1, keepdim=True) / denominator)


def _context(caption: Caption, model: QwenImageTransformer, device: torch.device) -> Tensor:
    """Validate and trim one native caption before denoising, preserving token order."""
    values, mask = caption.features, caption.mask
    if (
        values.ndim != 3
        or values.shape[0] != 1
        or values.shape[1] < 1
        or values.shape[2] != model.config.joint_attention_dim
        or not values.is_floating_point()
    ):
        raise ValueError("Qwen sampling requires one nonempty1LD caption with native feature width")
    if mask is not None:
        if mask.shape != values.shape[:2] or mask.dtype != torch.bool or mask.device != values.device:
            raise ValueError("Qwen caption mask must be matching boolean1L")
        values = values[:, mask[0]]
    if values.shape[1] == 0 or not torch.isfinite(values).all():
        raise ValueError("Qwen caption must contain finite real tokens")
    return values.to(device=device)


@torch.no_grad()
def sample_latents(
    model: QwenImageTransformer,
    noise: Tensor,
    positive: Caption,
    *,
    negative: Caption | None = None,
    references: tuple[Tensor, ...] = (),
    inpainting_mask: Tensor | None = None,
    steps: int = 25,
    shift: float | None = None,
    guidance: float = 4.0,
    allow_cpu_frozen: bool = False,
) -> Tensor:
    """Return normalized floating BCLHW image/layer latents from explicit Gaussian noise.

    One request per call; each ordered reference is normalized1C1HW with even
    spatial extents. Layered noise includes composite plus foreground layers and
    requires one clean source. Original image accepts no references; edit requires
    at least one. CFG runs whenever guidance differs from1, including scales0
    and below1, and normalizes each packed patch to its positive prediction norm.

    Optional edit mask is1N1 in packed patch order, white for regenerated tokens;
    the first reference must match output geometry. Blend the current state with
    the noised reference after prediction, before Euler, preserving native order.
    Schedule coefficients retain their native precision; state retains the
    caller's noise dtype. Preview compute follows training precision without
    changing model weights, RNG, input storage, mode or residency.
    """
    if (
        noise.ndim != 5
        or noise.shape[0] != 1
        or min(noise.shape) < 1
        or noise.shape[1] != model.config.out_channels
        or noise.dtype not in FLOAT_DTYPES.values()
        or any(value % 2 for value in noise.shape[-2:])
        or not torch.isfinite(noise).all()
    ):
        raise ValueError("Qwen sampling requires finite floating nonempty1CLHW noise with even H/W")
    if any(
        (parameter.is_floating_point() and parameter.dtype == torch.float64)
        or (
            parameter.device != noise.device
            and not (allow_cpu_frozen and parameter.device.type == "cpu" and not parameter.requires_grad)
        )
        for parameter in model.parameters()
    ):
        raise ValueError("Qwen sampling requires supported weights on the noise device")
    if not math.isfinite(guidance):
        raise ValueError("Qwen guidance must be finite")
    layered = model.config.variant == "layered"
    edit = model.config.variant.startswith("edit")
    if (not layered and noise.shape[2] != 1) or (not edit and not layered and references):
        raise ValueError("Qwen original/edit output has one layer; original consumes no references")
    if (edit and not references) or (layered and len(references) != 1):
        raise ValueError("Qwen edit/layered sampling requires its ordered reference latents")
    grids = [(1, noise.shape[-2] // 2, noise.shape[-1] // 2)] * noise.shape[2]
    for reference in references:
        if (
            reference.ndim != 5
            or reference.shape[:3] != (1, noise.shape[1], 1)
            or min(reference.shape) < 1
            or any(value % 2 for value in reference.shape[-2:])
            or reference.device != noise.device
            or reference.dtype not in FLOAT_DTYPES.values()
            or not torch.isfinite(reference).all()
        ):
            raise ValueError("Qwen references require finite normalized floating 1C1HW with even H/W on noise device")
        grids.append((1, reference.shape[-2] // 2, reference.shape[-1] // 2))
    references = tuple(value.to(noise.dtype) for value in references)
    state = pack_latents(noise).clone()
    initial = state
    fixed = None if not references else torch.cat([pack_latents(value) for value in references], dim=1)
    if inpainting_mask is not None and (
        not edit
        or not references
        or references[0].shape[-2:] != noise.shape[-2:]
        or inpainting_mask.shape != (1, state.shape[1], 1)
        or inpainting_mask.device != noise.device
        or inpainting_mask.dtype not in FLOAT_DTYPES.values()
        or not torch.isfinite(inpainting_mask).all()
        or torch.any((inpainting_mask < 0) | (inpainting_mask > 1))
    ):
        raise ValueError("Qwen mask requires an edit reference matching output geometry and finite1N1 weights in[0,1]")
    if inpainting_mask is not None:
        inpainting_mask = inpainting_mask.to(noise.dtype)
    source = pack_latents(references[0]) if inpainting_mask is not None else None
    context = _context(positive, model, noise.device).to(noise.dtype)
    if guidance != 1 and negative is None:
        raise ValueError("Qwen CFG requires a negative caption when guidance differs from one")
    negative_context = None if negative is None else _context(negative, model, noise.device).to(noise.dtype)
    count = pack_latents(references[0]).shape[1] if layered else state.shape[1]
    schedule = build_schedule(steps, count, shift=shift, layered=layered).to(noise.device)
    times = schedule[:-1] * 1000
    condition = torch.zeros(1, dtype=torch.int64, device=noise.device) if layered else None
    with sampling_context(model, noise.device, state_dtype=noise.dtype):
        for index, time in enumerate(times):
            tokens = state if fixed is None else torch.cat((state, fixed), dim=1)
            timestep = time.expand(1) / 1000
            prediction = model(tokens, context, timestep, tuple(grids), layer_condition=condition)[:, : state.shape[1]]
            if guidance != 1:
                if negative_context is None:
                    raise RuntimeError("Qwen negative context was not assembled")
                unconditional = model(tokens, negative_context, timestep, tuple(grids), layer_condition=condition)[
                    :, : state.shape[1]
                ]
                prediction = normalized_guidance(prediction, unconditional, guidance)
            if inpainting_mask is not None:
                fraction = time / 1000
                if source is None:
                    raise RuntimeError("Qwen masked source was not assembled")
                noised = source * (1 - fraction) + initial * fraction
                state = state * inpainting_mask + noised * (1.0 - inpainting_mask)
            state = state + (schedule[index + 1] - schedule[index]) * prediction.to(noise.dtype)
    return unpack_latents(state.to(noise.dtype), layers=noise.shape[2], height=noise.shape[-2], width=noise.shape[-1])


@torch.no_grad()
def generate_latents(
    model: QwenImageTransformer,
    positive: Caption,
    sampling: SamplingConfig,
    *,
    negative: Caption | None = None,
    vae_factor: int | None = None,
    condition: FluxCondition | None = None,
    references: tuple[Tensor, ...] = (),
    inpainting_mask: Tensor | None = None,
    allow_cpu_frozen: bool = False,
) -> Tensor:
    """Initialize native Qwen Image noise and sample without changing model mode.

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
        Model-native BCLHW layers latents on the model device without a gradient graph.
        Seed-local noise draw order, BF16 rounding where native, solver defaults
        and input tensors remain unchanged; no global RNG is advanced.

    Raises:
        ValueError: Features, conditions or dimensions violate the native contract.
    """
    validate_image_sampling(sampling, family="qwen-image", variant=model.config.variant, version=None)
    if isinstance(positive, FluxTextFeatures):
        raise ValueError("Flux requires T5/CLIP features; other image models require Caption")
    if condition is not None:
        raise ValueError("Packed image conditions require Flux")
    if vae_factor is None:
        vae_factor = 8
    granularity = vae_factor * 2
    if vae_factor < 1 or sampling.height % granularity or sampling.width % granularity:
        raise ValueError("Sampling dimensions must match vae factor and spatial patch size")
    parameter = next(model.parameters())
    generator = torch.Generator(device="cpu").manual_seed(sampling.seed)
    if not isinstance(positive, Caption):
        raise TypeError("Image model requires Caption")
    layers = (
        (4 if sampling.output_layers is None else sampling.output_layers) + 1
        if model.config.variant == "layered"
        else 1
    )
    noise = (
        torch.randn(
            (1, layers, model.config.out_channels, sampling.height // vae_factor, sampling.width // vae_factor),
            generator=generator,
            device="cpu",
            dtype=torch.bfloat16,
        )
        .transpose(1, 2)
        .float()
        .to(device=parameter.device, dtype=sampling_dtype(model))
    )
    return sample_latents(
        model,
        noise,
        positive,
        negative=negative,
        references=tuple(value.to(device=parameter.device, dtype=torch.float32) for value in references),
        inpainting_mask=None if inpainting_mask is None else inpainting_mask.to(device=parameter.device),
        allow_cpu_frozen=allow_cpu_frozen,
        steps=25 if sampling.steps is None else sampling.steps,
        shift=sampling.shift,
        guidance=4.0 if sampling.guidance is None else sampling.guidance,
    )
