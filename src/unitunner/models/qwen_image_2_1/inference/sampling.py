# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Integrate native Qwen2.1 target velocities with plain CFG and image-level masks."""

import math

import torch
from torch import Tensor

from unitunner.config.sampling import SamplingConfig, validate_image_sampling
from unitunner.models.flux.conditioning import FluxCondition
from unitunner.models.qwen_image.inference.sampling import build_schedule
from unitunner.models.qwen_image_2_1.conditioning import QwenImage21TextFeatures, pack_latents, unpack_latents
from unitunner.models.qwen_image_2_1.conditioning import unpack_latents as unpack_qwen21_latents
from unitunner.models.qwen_image_2_1.transformer import QwenImage21Transformer


@torch.no_grad()
def sample_latents(
    model: QwenImage21Transformer,
    noise: Tensor,
    positive: QwenImage21TextFeatures,
    *,
    negative: QwenImage21TextFeatures | None = None,
    references: tuple[Tensor, ...] = (),
    inpainting_mask: Tensor | None = None,
    steps: int = 25,
    shift: float | None = None,
    guidance: float = 1.0,
) -> Tensor:
    """Return FP32 1C1HW latents from explicit noise without mode/RNG/residency changes.

    Use native terminal-stretched Qwen sigmas; a one-step request is exactly
    [1,0] with no terminal stretching. Model time follows sigma*1000/1000.
    Plain CFG runs only above1, with separate positive/negative insertion slots;
    no previous-Qwen norm rescaling applies. Reference latents are independent
    normalized1C1HW images. Optional FP32 1N1 mask uses white for repaint and
    blends the first reference at the next sigma, ending with clean reference.
    """
    if (
        noise.ndim != 5
        or noise.shape[:3] != (1, model.config.in_channels, 1)
        or min(noise.shape) < 1
        or noise.dtype != torch.float32
        or model.config.out_channels != noise.shape[1]
        or not torch.isfinite(noise).all()
    ):
        raise ValueError("Qwen Image2.1 sampling requires finite FP32 1C1HW native-channel noise")
    parameter = next(model.parameters())
    if parameter.device != noise.device or parameter.dtype != torch.float32:
        raise ValueError("Qwen Image2.1 sampling requires a matching FP32 model device")
    if type(steps) is not int or steps < 1 or not math.isfinite(guidance):
        raise ValueError("Qwen Image2.1 sampling requires positive integer steps and finite guidance")
    if shift is not None and (not math.isfinite(shift) or shift <= 0 or shift > torch.finfo(torch.float32).max):
        raise ValueError("Qwen Image2.1 shift must be positive finite FP32")
    for reference in references:
        if (
            reference.ndim != 5
            or reference.shape[:3] != noise.shape[:3]
            or min(reference.shape) < 1
            or reference.dtype != noise.dtype
            or reference.device != noise.device
            or not torch.isfinite(reference).all()
        ):
            raise ValueError("Qwen Image2.1 references must be finite matching1C1HW with independent positive grids")
    shapes = tuple((reference.shape[-2], reference.shape[-1]) for reference in references)
    for prompt in (positive, negative) if guidance > 1 else (positive,):
        if prompt is None:
            raise ValueError("Qwen Image2.1 CFG above1 requires a negative prompt")
        if (
            prompt.features.ndim != 2
            or min(prompt.features.shape) < 1
            or prompt.features.shape[1] != model.config.context_in_dim
            or not prompt.features.is_floating_point()
            or not torch.isfinite(prompt.features).all()
            or prompt.image_shapes != shapes
            or len(prompt.image_slots) != len(references)
            or any(type(slot) is not int or not 0 <= slot <= prompt.features.shape[0] for slot in prompt.image_slots)
            or tuple(sorted(prompt.image_slots)) != prompt.image_slots
        ):
            raise ValueError("Qwen Image2.1 prompt features, reference grids and integer insertion slots must agree")
    count = noise.shape[-2] * noise.shape[-1]
    if inpainting_mask is not None:
        if (
            not references
            or references[0].shape != noise.shape
            or inpainting_mask.shape != (1, count, 1)
            or inpainting_mask.dtype != noise.dtype
            or inpainting_mask.device != noise.device
            or not torch.isfinite(inpainting_mask).all()
            or torch.any(inpainting_mask < 0)
            or torch.any(inpainting_mask > 1)
        ):
            raise ValueError(
                "Qwen Image2.1 inpainting requires a matching first reference and finite FP32 1N1 mask in[0,1]"
            )
    schedule = (
        torch.tensor([1.0, 0.0], device="cpu", dtype=torch.float32)
        if steps == 1
        else build_schedule(steps, count, shift=shift)
    ).to(noise.device)
    times = schedule[:-1] * 1000
    reference_tokens = tuple(pack_latents(reference) for reference in references)
    grids = (*((1, height, width) for height, width in shapes), (1, noise.shape[-2], noise.shape[-1]))
    initial = pack_latents(noise)
    state = initial
    positive_context = positive.features.to(device=noise.device, dtype=torch.float32)[None]
    negative_context = (
        negative.features.to(device=noise.device, dtype=torch.float32)[None]
        if guidance > 1 and negative is not None
        else None
    )

    def predict(context: Tensor, slots: tuple[int, ...], time: Tensor) -> Tensor:
        """Predict target velocity with contexts already staged on the model device."""
        return model(
            state,
            context,
            time.reshape(1) / 1000,
            (grids,),
            references=reference_tokens,
            image_slots=(slots,),
        )

    with torch.autocast(noise.device.type, enabled=False):
        for index, time in enumerate(times):
            prediction = predict(positive_context, positive.image_slots, time)
            if guidance > 1:
                if negative is None or negative_context is None:
                    raise ValueError("Qwen Image2.1 CFG requires negative prompt")
                unconditioned = predict(negative_context, negative.image_slots, time)
                prediction = unconditioned + guidance * (prediction - unconditioned)
            state = state + (schedule[index + 1] - schedule[index]) * prediction
            if inpainting_mask is not None:
                sigma = schedule[index + 1]
                control = reference_tokens[0] * (1 - sigma) + initial * sigma
                state = state * inpainting_mask + control * (1 - inpainting_mask)
    return unpack_latents(state, height=noise.shape[-2], width=noise.shape[-1])


@torch.no_grad()
def generate_latents(
    model: QwenImage21Transformer,
    positive: QwenImage21TextFeatures,
    sampling: SamplingConfig,
    *,
    negative: QwenImage21TextFeatures | None = None,
    vae_factor: int | None = None,
    condition: FluxCondition | None = None,
    references: tuple[Tensor, ...] = (),
    inpainting_mask: Tensor | None = None,
) -> Tensor:
    """Initialize native Qwen Image 2.1 noise and sample without changing model mode.

    Args:
        model: Borrowed native transformer; the caller owns eval mode and residency.
        positive: Native unpadded LD features with host insertion slots and reference shapes.
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
        Model-native BC1HW latents on the model device without a gradient graph.
        Seed-local noise draw order, BF16 rounding where native, solver defaults
        and input tensors remain unchanged; no global RNG is advanced.

    Raises:
        ValueError: Features, conditions or dimensions violate the native contract.
    """
    validate_image_sampling(sampling, family="qwen-image-2-1", variant="base")
    if not isinstance(positive, QwenImage21TextFeatures) or (
        negative is not None and (not isinstance(negative, QwenImage21TextFeatures))
    ):
        raise ValueError("Qwen Image2.1 requires native text features with explicit reference insertion slots")
    if condition is not None or vae_factor not in (None, 16):
        raise ValueError("Qwen Image2.1 requires native factor16 vae and ordered reference images")
    parameter = next(model.parameters())
    height, width = (sampling.height // 16, sampling.width // 16)
    generator = torch.Generator(device="cpu").manual_seed(sampling.seed)
    tokens = torch.randn(
        (1, height * width, model.config.in_channels), generator=generator, device="cpu", dtype=torch.float32
    )
    return sample_latents(
        model,
        unpack_qwen21_latents(tokens, height=height, width=width).to(parameter.device),
        positive,
        negative=negative,
        references=tuple(value.to(parameter.device) for value in references),
        inpainting_mask=None if inpainting_mask is None else inpainting_mask.to(parameter.device),
        steps=25 if sampling.steps is None else sampling.steps,
        shift=sampling.shift,
        guidance=1.0 if sampling.guidance is None else sampling.guidance,
    )
