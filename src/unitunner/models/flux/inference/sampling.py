"""Integrate native FLUX.1 velocity with explicit noise and variant conditions."""

import math
from itertools import pairwise

import torch
from torch import Tensor

from unitunner.config.sampling import SamplingConfig, validate_image_sampling
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux.conditioning import (
    FluxCondition,
    FluxTextFeatures,
    build_inputs,
    image_ids,
    pack_latents,
    unpack_latents,
)
from unitunner.models.flux.transformer import FluxTransformer
from unitunner.precision import FLOAT_DTYPES, sampling_context, sampling_dtype


def build_schedule(steps: int, token_count: int, *, shift: float | None = None) -> tuple[float, ...]:
    """Return native CPU FP32-derived noise fractions descending from one to zero.

    An explicit positive shift uses s*t/(1+(s-1)*t). Otherwise the exponential
    multiplier uses the native linear resolution shift: 0.5 at 256 tokens and
    1.15 at 4096 tokens, extrapolated outside these endpoints. Return host values
    to preserve Euler's original Python timestep subtraction before multiplication.
    No global default dtype/device or RNG state is used.
    """
    if type(steps) is not int or type(token_count) is not int or min(steps, token_count) < 1:
        raise ValueError("FLUX schedule requires positive integer steps and token_count")
    grid = torch.linspace(1, 0, steps + 1, dtype=torch.float32, device="cpu")
    if shift is None:
        slope = (1.15 - 0.5) / (4096 - 256)
        mu = slope * token_count + (0.5 - slope * 256)
        if not math.isfinite(mu) or mu > math.log(torch.finfo(torch.float32).max):
            raise ValueError("FLUX resolution shift exceeds finite FP32 range")
        multiplier = math.exp(mu)
        grid = multiplier / (multiplier + (1 / grid - 1) ** 1.0)
    else:
        if not math.isfinite(shift) or shift <= 0 or shift > torch.finfo(torch.float32).max:
            raise ValueError("FLUX shift must be finite, positive and FP32-representable")
        grid = shift * grid / (1 + (shift - 1) * grid)
    if not torch.isfinite(grid).all() or grid[0] != 1 or grid[-1] != 0 or not torch.all(grid[:-1] > grid[1:]):
        raise ValueError("FLUX schedule must be finite and strictly decrease from one to zero")
    return tuple(grid.tolist())


@torch.no_grad()
def sample_latents(
    model: FluxTransformer,
    noise: Tensor,
    captions: Tensor,
    pooled: Tensor,
    *,
    condition: FluxCondition | None = None,
    steps: int = 25,
    shift: float | None = None,
    guidance: float | None = None,
    allow_cpu_frozen: bool = False,
) -> Tensor:
    """Return model-normalized floating BCHW latents from caller-supplied noise.

    Guidance is a native learned scalar embedding, with Dev/Kontext/Fill defaults
    3.5/2.5/30; it is not classifier-free guidance. Explicit conditions use packed
    reference tokens or Fill masked-image/mask features. Keep references fixed and
    update only target tokens. The caller owns model mode, weights and RNG;
    compute follows preview policy or resident floating dtype. State retains the
    caller's FP32/FP16/BF16 noise dtype, including with quantized base storage.
    """
    if noise.dtype not in FLOAT_DTYPES.values():
        raise ValueError("FLUX sampling requires FP32, BF16 or FP16 noise")
    if any(
        (parameter.is_floating_point() and parameter.dtype == torch.float64)
        or (
            parameter.device != noise.device
            and not (allow_cpu_frozen and parameter.device.type == "cpu" and not parameter.requires_grad)
        )
        for parameter in model.parameters()
    ):
        raise ValueError("FLUX sampling requires supported weights on the noise device")
    packed = pack_latents(noise)
    if condition is not None:
        condition = FluxCondition(
            condition.tokens.to(noise.dtype),
            condition.ids,
            condition.lengths,
            None if condition.fill_mask is None else condition.fill_mask.to(noise.dtype),
        )
    if packed.shape[2] != model.config.out_channels:
        raise ValueError("FLUX noise channels differ from transformer output width")
    if model.config.guidance_embed:
        if guidance is None:
            guidance = {"dev": 3.5, "kontext": 2.5, "fill": 30.0}[model.config.variant]
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("FLUX guidance must be finite and nonnegative")
        guidance_vector = torch.full((noise.shape[0],), guidance, dtype=torch.float32, device=noise.device)
    else:
        if guidance is not None:
            raise ValueError("FLUX topology without guidance embedding cannot consume guidance")
        guidance_vector = None
    ids = image_ids(noise.shape[0], noise.shape[2] // 2, noise.shape[3] // 2, device=noise.device)
    txt_ids = torch.zeros((*captions.shape[:2], 3), dtype=torch.float32, device=noise.device)
    schedule = build_schedule(steps, packed.shape[1], shift=shift)
    with sampling_context(model, noise.device, state_dtype=noise.dtype):
        for current, previous in pairwise(schedule):
            inputs = build_inputs(model.config.variant, packed, ids, condition)
            timesteps = torch.full((noise.shape[0],), current, dtype=torch.float32, device=noise.device)
            prediction = model(
                inputs.tokens,
                inputs.ids,
                captions,
                txt_ids,
                timesteps,
                pooled,
                guidance_vector,
                control_lengths=inputs.control_lengths,
            )
            packed = packed + (previous - current) * prediction.to(noise.dtype)[:, : packed.shape[1]]
    return unpack_latents(packed, noise.shape[2], noise.shape[3])


@torch.no_grad()
def generate_latents(
    model: FluxTransformer,
    positive: FluxTextFeatures,
    sampling: SamplingConfig,
    *,
    negative: Caption | None = None,
    vae_factor: int | None = None,
    condition: FluxCondition | None = None,
    references: tuple[Tensor, ...] = (),
    inpainting_mask: Tensor | None = None,
    allow_cpu_frozen: bool = False,
) -> Tensor:
    """Initialize native FLUX noise and sample.

    Return native latents on the model device using seed-local CPU noise.
    The caller owns eval mode and model residency.
    """
    validate_image_sampling(sampling, family="flux", variant=model.config.variant, version=None)
    if not isinstance(positive, FluxTextFeatures):
        raise ValueError("Flux requires T5/CLIP features; other image models require Caption")
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
    if not isinstance(positive, FluxTextFeatures) or negative is not None:
        raise ValueError("Flux sampling requires native T5/CLIP features without negative conditioning")
    height, width = (sampling.height // vae_factor, sampling.width // vae_factor)
    if condition is not None:
        condition = FluxCondition(
            condition.tokens.to(device=parameter.device, dtype=sampling_dtype(model)),
            None if condition.ids is None else condition.ids.to(device=parameter.device),
            condition.lengths,
            None
            if condition.fill_mask is None
            else condition.fill_mask.to(device=parameter.device, dtype=sampling_dtype(model)),
        )
    noise = unpack_latents(
        torch.randn(
            (1, height * width // 4, model.config.out_channels), generator=generator, device="cpu", dtype=torch.float32
        ),
        height,
        width,
    )
    return sample_latents(
        model,
        noise.to(device=parameter.device, dtype=sampling_dtype(model)),
        positive.captions.to(device=parameter.device, dtype=sampling_dtype(model)),
        positive.pooled.to(device=parameter.device, dtype=sampling_dtype(model)),
        condition=condition,
        allow_cpu_frozen=allow_cpu_frozen,
        steps=25 if sampling.steps is None else sampling.steps,
        shift=sampling.shift,
        guidance=sampling.guidance,
    )
