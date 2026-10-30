"""Integrate FLUX.2 Dev embedded guidance or Klein CFG with native empirical time shifts."""

import math
from itertools import pairwise

import torch
from torch import Tensor

from unitunner.config.sampling import SamplingConfig, validate_image_sampling
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux.conditioning import FluxCondition, FluxTextFeatures
from unitunner.models.flux2.conditioning import build_image_inputs, pack_text, unpack_image
from unitunner.models.flux2.transformer import Flux2Transformer
from unitunner.precision import FLOAT_DTYPES, sampling_context, sampling_dtype


def build_schedule(steps: int, token_count: int, *, shift: float | None = None) -> tuple[float, ...]:
    """Return the native empirical CPU FP32 Euler grid descending to zero.

    For at most4300 target tokens, interpolate the ten/200-step fitted log-SNR
    shifts using the actual step count. Above4300 use the200-step fit directly.
    Explicit positive shift uses s*t/(1+(s-1)*t). Preserve native host timestep
    subtraction; pin dtype/device and reject collapsed/nonfinite grids. The
    empirical division may round its first value one FP32 ulp away from one;
    preserve that reference value instead of silently snapping it.
    """
    if type(steps) is not int or type(token_count) is not int or min(steps, token_count) < 1:
        raise ValueError("FLUX.2 schedule requires positive integer steps and token_count")
    grid = torch.linspace(1, 0, steps + 1, dtype=torch.float32, device="cpu")
    if shift is None:
        m200 = 0.00016927 * token_count + 0.45666666
        m10 = 8.73809524e-05 * token_count + 1.89833333
        slope = (m200 - m10) / 190.0
        mu = m200 if token_count > 4300 else slope * steps + (m200 - 200.0 * slope)
        if not math.isfinite(mu) or mu > math.log(torch.finfo(torch.float32).max):
            raise ValueError("FLUX.2 empirical shift exceeds finite FP32 range")
        multiplier = math.exp(mu)
        grid = multiplier / (multiplier + (1 / grid - 1) ** 1.0)
    else:
        if not math.isfinite(shift) or shift <= 0 or shift > torch.finfo(torch.float32).max:
            raise ValueError("FLUX.2 shift must be finite, positive and FP32-representable")
        grid = (grid * shift) / (1 + (shift - 1) * grid)
    if (
        not torch.isfinite(grid).all()
        or abs(float(grid[0]) - 1) > torch.finfo(torch.float32).eps
        or grid[-1] != 0
        or not (grid[:-1] > grid[1:]).all()
    ):
        raise ValueError("FLUX.2 schedule must strictly decrease from one to zero")
    return tuple(grid.tolist())


@torch.no_grad()
def sample_latents(
    model: Flux2Transformer,
    noise: Tensor,
    captions: tuple[Tensor, ...],
    *,
    negative: tuple[Tensor, ...] | None = None,
    references: tuple[tuple[Tensor, ...], ...] = (),
    steps: int | None = None,
    shift: float | None = None,
    guidance: float | None = None,
    allow_cpu_frozen: bool = False,
) -> Tensor:
    """Return floating BCHW packed-VAE latents from explicit noise and LD captions.

    Dev uses embedded guidance(default4) and50 steps. Klein base requires negative
    context and external CFG(default4) with50 steps; distilled Klein defaults to
    four steps and consumes neither negative context nor embedded guidance.
    Reference CHW tensors remain fixed, with native time offsets10,20,... .
    Samples execute independently without inventing attention padding. Text keeps
    every supplied padding token. Caller owns weights, modes, RNG and any initial
    native value rounding. Preview compute follows training precision; state
    retains the noise dtype and inputs remain unchanged.
    """
    if (
        noise.ndim != 4
        or min(noise.shape) < 1
        or noise.dtype not in FLOAT_DTYPES.values()
        or noise.shape[1] != model.config.in_channels
        or len(captions) != noise.shape[0]
        or (references and len(references) != noise.shape[0])
    ):
        raise ValueError("FLUX.2 sampling requires floating BCHW noise and one caption/reference row per sample")
    if any(
        (p.is_floating_point() and p.dtype == torch.float64)
        or (p.device != noise.device and not (allow_cpu_frozen and p.device.type == "cpu" and not p.requires_grad))
        for p in model.parameters()
    ):
        raise ValueError("FLUX.2 sampling requires supported weights on the noise device")
    cfg = model.config.variant == "base"
    if cfg:
        if negative is None or len(negative) != noise.shape[0]:
            raise ValueError("FLUX.2 Klein base requires one negative caption per sample")
    elif negative is not None:
        raise ValueError("FLUX.2 distilled checkpoints do not consume negative captions")
    if not model.config.use_guidance_embed and not cfg:
        if guidance not in (None, 1.0):
            raise ValueError("FLUX.2 distilled Klein requires guidance1 without an embedded branch")
        guidance = 1.0
    else:
        guidance = 4.0 if guidance is None else guidance
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("FLUX.2 guidance must be finite and nonnegative")
    schedule = build_schedule(
        (4 if not cfg and not model.config.use_guidance_embed else 50) if steps is None else steps,
        noise.shape[2] * noise.shape[3],
        shift=shift,
    )
    rows = []
    with sampling_context(model, noise.device, state_dtype=noise.dtype):
        for index, initial in enumerate(noise):
            image = build_image_inputs(
                initial, tuple(value.to(noise.dtype) for value in references[index]) if references else ()
            )
            context, context_ids = pack_text(captions[index])
            if context.device != noise.device or context.dtype not in FLOAT_DTYPES.values():
                raise ValueError("FLUX.2 captions must be floating on the noise device")
            negative_context = negative_ids = None
            if negative is not None:
                negative_context, negative_ids = pack_text(negative[index])
                if negative_context.device != noise.device or negative_context.dtype not in FLOAT_DTYPES.values():
                    raise ValueError("FLUX.2 negative captions must be floating on the noise device")
            state = image.tokens[:, : image.target_count].clone()
            clean = image.tokens[:, image.target_count :]
            embedded = state.new_full((1,), guidance) if model.config.use_guidance_embed else None
            for current, following in pairwise(schedule):
                tokens = torch.cat((state, clean), dim=1)
                time = state.new_full((1,), current)
                prediction = model(tokens, image.ids, time, context.unsqueeze(0), context_ids.unsqueeze(0), embedded)
                if negative_context is not None and negative_ids is not None:
                    unconditional = model(
                        tokens, image.ids, time, negative_context.unsqueeze(0), negative_ids.unsqueeze(0), None
                    )
                    prediction = unconditional + guidance * (prediction - unconditional)
                state = state + (following - current) * prediction.to(noise.dtype)[:, : image.target_count]
            rows.append(unpack_image(state[0], height=noise.shape[2], width=noise.shape[3]))
    return torch.stack(rows)


@torch.no_grad()
def generate_latents(
    model: Flux2Transformer,
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
    """Initialize native FLUX.2 noise and sample.

    Return native latents on the model device using seed-local CPU noise.
    The caller owns eval mode and model residency.
    """
    validate_image_sampling(sampling, family="flux2", variant=model.config.variant, version=model.config.version)
    if isinstance(positive, FluxTextFeatures):
        raise ValueError("Flux requires T5/CLIP features; other image models require Caption")
    if condition is not None:
        raise ValueError("Packed image conditions require Flux")
    if inpainting_mask is not None:
        raise ValueError("Packed inpainting masks require Qwen Image")
    if vae_factor is None:
        vae_factor = 16
    granularity = vae_factor
    if vae_factor < 1 or sampling.height % granularity or sampling.width % granularity:
        raise ValueError("Sampling dimensions must match vae factor and spatial patch size")
    parameter = next(model.parameters())
    generator = torch.Generator(device="cpu").manual_seed(sampling.seed)
    if not isinstance(positive, Caption):
        raise TypeError("Image model requires Caption")
    if (
        positive.features.shape[0] != 1
        or positive.mask is not None
        or (negative is not None and (negative.features.shape[0] != 1 or negative.mask is not None))
    ):
        raise ValueError("FLUX.2 generation requires one caption retaining all encoder padding")
    noise = (
        torch.randn(
            (1, model.config.in_channels, sampling.height // vae_factor, sampling.width // vae_factor),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        .to(torch.bfloat16)
        .float()
    )
    return sample_latents(
        model,
        noise.to(device=parameter.device, dtype=sampling_dtype(model)),
        (positive.features[0].to(device=parameter.device, dtype=torch.float32),),
        negative=None if negative is None else (negative.features[0].to(device=parameter.device, dtype=torch.float32),),
        references=(tuple(value.to(device=parameter.device, dtype=torch.float32) for value in references),),
        allow_cpu_frozen=allow_cpu_frozen,
        steps=sampling.steps,
        shift=sampling.shift,
        guidance=sampling.guidance,
    )
