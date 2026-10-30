# Derived from Black Forest Labs FLUX3 Image Klein, Apache-2.0.
"""Integrate FLUX3 base CFG or distilled flow maps on the official shifted Euler grid."""

import math
from dataclasses import replace

import torch
from torch import Tensor

from unituner.config.sampling import SamplingConfig, validate_image_sampling
from unituner.models.components.conditioning import Caption
from unituner.models.flux.conditioning import FluxCondition, FluxTextFeatures
from unituner.models.flux3.conditioning import Flux3Sample, image_tokens, pack_samples, reference_tokens, text_tokens
from unituner.models.flux3.transformer import Flux3Transformer
from unituner.precision import FLOAT_DTYPES, sampling_context, sampling_dtype


def build_schedule(steps: int, *, shift: float = 16.0) -> Tensor:
    """Return descending CPU FP32 noise fractions including one and zero.

    Use the official rational shift without resolution-dependent extrapolation or
    training-grid snapping. Pin dtype/device independently of process defaults;
    reject collapsed/nonfinite schedules instead of running zero-length steps.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("FLUX3 steps must be a positive integer")
    if not math.isfinite(shift) or shift <= 0 or shift > torch.finfo(torch.float32).max:
        raise ValueError("FLUX3 shift must be finite, positive and FP32-representable")
    values = torch.linspace(1, 0, steps + 1, dtype=torch.float32, device="cpu")
    values = shift * values / (1 + (shift - 1) * values)
    if not torch.isfinite(values).all() or values[0] != 1 or values[-1] != 0 or not (values[:-1] > values[1:]).all():
        raise ValueError("FLUX3 schedule must strictly decrease from one to zero")
    return values


@torch.no_grad()
def sample_latents(
    model: Flux3Transformer,
    noise: Tensor,
    captions: tuple[Tensor, ...],
    *,
    negative: tuple[Tensor, ...] | None = None,
    references: tuple[tuple[Tensor, ...], ...] = (),
    steps: int | None = None,
    shift: float = 16.0,
    guidance: float | None = None,
    allow_cpu_frozen: bool = False,
) -> Tensor:
    """Return floating BCHW packed-VAE latents using explicit noise and LD captions.

    Base uses [negative,positive] packed samples and unconditional + guidance *
    (conditional-unconditional), defaulting to 50 steps and guidance 4.0.
    Distilled uses 16 steps with the next schedule value as its destination time;
    it accepts neither negative features nor guidance. Ordered clean CHW reference
    latents are fixed throughout sampling, with independent geometry per image.

    The caller owns model mode, VAE, text encoding and Gaussian noise policy.
    No input storage, module mode or RNG is changed. Sigma and Euler accumulation
    retain the caller's floating noise dtype. Transformer compute follows the
    scoped preview policy or resident floating dtype, including quantized bases.
    Native timestep rounding remains inside the model boundary.
    """
    if (
        noise.ndim != 4
        or min(noise.shape) < 1
        or noise.dtype not in FLOAT_DTYPES.values()
        or noise.shape[1] != model.config.image_channels
        or len(captions) != noise.shape[0]
        or (references and len(references) != noise.shape[0])
    ):
        raise ValueError(
            "FLUX3 sampling requires nonempty floating BCHW noise and one caption/reference row per sample"
        )
    if any(
        (parameter.is_floating_point() and parameter.dtype == torch.float64)
        or (
            parameter.device != noise.device
            and not (allow_cpu_frozen and parameter.device.type == "cpu" and not parameter.requires_grad)
        )
        for parameter in model.parameters()
    ):
        raise ValueError("FLUX3 sampling requires supported weights on the noise device")
    distilled = model.config.variant == "distilled"
    if distilled:
        if guidance is not None or negative is not None:
            raise ValueError("FLUX3 distilled flow maps do not consume CFG guidance or negative captions")
    else:
        if negative is None or len(negative) != noise.shape[0]:
            raise ValueError("FLUX3 base CFG requires one negative caption per sample")
        guidance = 4.0 if guidance is None else guidance
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("FLUX3 CFG guidance must be finite and nonnegative")
    schedule = build_schedule((16 if distilled else 50) if steps is None else steps, shift=shift)
    positive = [
        Flux3Sample(
            image_tokens(value),
            text_tokens(captions[index].to(noise.dtype)),
            reference_tokens(tuple(value.to(noise.dtype) for value in references[index])) if references else None,
        )
        for index, value in enumerate(noise)
    ]
    samples = positive
    if negative is not None:
        samples = [
            replace(sample, text=text_tokens(negative[index].to(noise.dtype))) for index, sample in enumerate(positive)
        ] + positive
    packed = pack_samples(samples, torch.zeros(len(samples), device=noise.device, dtype=torch.float32))
    state = torch.cat([sample.image.values for sample in positive]).clone()
    # Subtract on the native FP32 schedule before conversion to host scalars.
    deltas = (schedule[1:] - schedule[:-1]).tolist()
    with sampling_context(model, noise.device, state_dtype=noise.dtype):
        for current, following, delta in zip(schedule[:-1].tolist(), schedule[1:].tolist(), deltas, strict=True):
            packed = replace(
                packed,
                data={**packed.data, "image": state if distilled else torch.cat((state, state))},
                timesteps={**packed.timesteps, "image": torch.full_like(packed.timesteps["image"], current)},
            )
            destination = state.new_full((noise.shape[0],), following) if distilled else None
            prediction = model(packed, destination)
            if not distilled:
                unconditional, conditional = prediction.chunk(2)
                prediction = unconditional + guidance * (conditional - unconditional)
            state = state + delta * prediction.to(noise.dtype)
    return state.reshape(noise.shape[0], noise.shape[2], noise.shape[3], noise.shape[1]).permute(0, 3, 1, 2)


@torch.no_grad()
def generate_latents(
    model: Flux3Transformer,
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
    """Initialize native FLUX3 noise and sample.

    Return native latents on the model device using seed-local CPU noise.
    The caller owns eval mode and model residency.
    """
    validate_image_sampling(sampling, family="flux3", variant=model.config.variant, version=None)
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
        raise ValueError("FLUX3 generation requires one caption retaining all encoder padding without a mask")
    height, width = (sampling.height // vae_factor, sampling.width // vae_factor)
    noise = (
        torch.randn(
            (height * width, model.config.image_channels), generator=generator, device="cpu", dtype=torch.float32
        )
        .to(torch.bfloat16)
        .float()
    )
    noise = noise.T.reshape(1, model.config.image_channels, height, width).to(
        device=parameter.device, dtype=sampling_dtype(model)
    )
    return sample_latents(
        model,
        noise,
        (positive.features[0].to(device=parameter.device, dtype=torch.float32),),
        negative=None if negative is None else (negative.features[0].to(device=parameter.device, dtype=torch.float32),),
        references=(tuple(value.to(device=parameter.device, dtype=torch.float32) for value in references),),
        allow_cpu_frozen=allow_cpu_frozen,
        steps=sampling.steps,
        shift=16.0 if sampling.shift is None else sampling.shift,
        guidance=sampling.guidance,
    )
