# Adapted from OpenMOSS MOSS-Image, Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""Integrate MOSS native velocities on its nearest-training-grid Euler schedule."""

import math

import torch
from torch import Tensor

from unitunner.config.sampling import SamplingConfig, validate_image_sampling
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux.conditioning import FluxCondition, FluxTextFeatures
from unitunner.models.moss_image.transformer import MossImageTransformer
from unitunner.precision import FLOAT_DTYPES, sampling_context, sampling_dtype


def build_schedule(steps: int, *, shift: float = 5.0, training_shift: float = 1.0, total: int = 1000) -> Tensor:
    """Return descending CPU FP32 noise fractions with terminal zero.

    Match official revision 1ceede80's V2 schedule: rationally shift the inference
    grid, snap to the closest shifted training value (ties use the lower sigma),
    then remove adjacent duplicates. The number of actual steps can be smaller
    than requested. Shift is applied once; native model times are sigma*1000.
    """
    if type(steps) is not int or steps < 1 or type(total) is not int or total < 2:
        raise ValueError("MOSS requires positive integer steps and at least two training times")
    if any(not math.isfinite(value) or value <= 0 for value in (shift, training_shift)):
        raise ValueError("MOSS schedule shifts must be finite and positive")

    def grid(count: int, offset: float) -> Tensor:
        lower = offset / (total + offset - 1)
        values = torch.linspace(1, lower, count, dtype=torch.float32, device="cpu")
        return offset * values / (1 + (offset - 1) * values)

    inference = grid(steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, inference).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((inference - training[lower]).abs() <= (training[upper] - inference).abs(), lower, upper)
    return torch.cat((torch.unique_consecutive(training[nearest]), torch.zeros(1, dtype=torch.float32)))


@torch.no_grad()
def sample_latents(
    model: MossImageTransformer,
    noise: Tensor,
    positive: Tensor,
    *,
    positive_mask: Tensor | None = None,
    negative: Tensor | None = None,
    negative_mask: Tensor | None = None,
    steps: int = 50,
    shift: float = 5.0,
    training_shift: float = 1.0,
    guidance: float = 3.0,
    uncond_skip_layers: tuple[int, ...] = (22,),
    slg_layers: tuple[int, ...] = (),
    slg_scale: float = 0.0,
    slg_start: float = 0.0,
    slg_stop: float = 1.0,
    references: tuple[Tensor, ...] = (),
    use_reference_cache: bool = True,
) -> Tensor:
    """Return normalized floating BCHW latents from explicit caller-owned Gaussian noise.

    Positive/negative BLD features may have different caption lengths and BL
    prefix masks. Features/masks move once to the model device. State/velocity
    accumulation retains the noise dtype; compute uses the scoped preview
    policy or resident floating dtype. No RNG is consumed or input storage mutated.
    The caller owns eval mode, residency, VAE and parameter installations.

    CFG is negative+guidance*(positive-negative). During the half-open SLG
    interval, negative prediction skips uncond_skip_layers. Active positive SLG
    adds scale*(positive-skipped); without CFG its base is skipped, matching the
    official model. Invalid skip indices fail before the first forward.
    """
    parameter = model.video_embedder.weight
    if model.training:
        raise ValueError("MOSS sampling requires eval mode")
    if (
        noise.ndim != 4
        or not noise.numel()
        or noise.dtype not in FLOAT_DTYPES.values()
        or noise.shape[1] != model.config.latent_channels
        or noise.device != parameter.device
    ):
        raise ValueError("MOSS sampling noise must be nonempty floating BCHW on the model device")
    if (
        not math.isfinite(guidance)
        or guidance < 0
        or not math.isfinite(slg_scale)
        or slg_scale < 0
        or not 0 <= slg_start <= slg_stop <= 1
    ):
        raise ValueError("Invalid MOSS guidance or SLG interval")
    for indices in (uncond_skip_layers, slg_layers):
        if any(type(index) is not int or not 0 <= index < len(model.layers) for index in indices):
            raise ValueError("MOSS sampling skip-layer index outside main blocks")
    if guidance != 1 and negative is None:
        raise ValueError("MOSS CFG requires encoded negative text")
    for features, mask in ((positive, positive_mask), (negative, negative_mask)):
        if features is None:
            if mask is not None:
                raise ValueError("MOSS caption mask requires features")
            continue
        if (
            features.ndim != 3
            or features.shape[0] != noise.shape[0]
            or features.shape[1] < 1
            or features.shape[2] != model.config.text_dim
            or not features.is_floating_point()
        ):
            raise ValueError("MOSS sampling captions require floating BLD features")
        if mask is not None:
            host = mask.detach().cpu()
            if host.shape != features.shape[:2] or not ((host == 0) | (host == 1)).all():
                raise ValueError("MOSS sampling masks require binary BL prefixes")
            lengths = host.bool().sum(1)
            if (lengths < 1).any() or not torch.equal(
                host.bool(), torch.arange(host.shape[1])[None] < lengths[:, None]
            ):
                raise ValueError("MOSS sampling masks require nonempty right-padded prefixes")
    positive = positive.to(device=noise.device, dtype=sampling_dtype(model))
    negative = None if negative is None else negative.to(device=noise.device, dtype=sampling_dtype(model))
    positive_mask = None if positive_mask is None else positive_mask.to(noise.device)
    negative_mask = None if negative_mask is None else negative_mask.to(noise.device)
    times = build_schedule(steps, shift=shift, training_shift=training_shift)
    native_times = (times[:-1] * 1000).to(noise.device)
    latents = noise.clone()
    references = tuple(value.to(device=noise.device, dtype=sampling_dtype(model)) for value in references)
    if bool(references) != (model.config.variant == "edit"):
        raise ValueError("MOSS edit requires references; base/sft do not accept them")
    with sampling_context(model, noise.device, state_dtype=noise.dtype):
        caches: dict[tuple[int, ...], tuple[tuple[Tensor, Tensor], ...]] = {}

        def predict(
            inputs: Tensor, time: Tensor, text: Tensor, mask: Tensor | None, skipped: tuple[int, ...] = ()
        ) -> Tensor:
            if references and use_reference_cache:
                key = tuple(sorted(set(skipped)))
                if key not in caches:
                    caches[key] = model._prepare_reference_kv(references, noise.shape[0], key)
                return model(inputs, time, text, mask, skip_layers=skipped, _reference_kv=caches[key]).float()
            return model(inputs, time, text, mask, skip_layers=skipped, references=references).float()

        for index in range(len(times) - 1):
            inputs = latents.to(sampling_dtype(model))
            active = slg_start <= index / (len(times) - 1) < slg_stop
            conditional = predict(inputs, native_times[index], positive, positive_mask)
            prediction = conditional
            if guidance != 1 and negative is not None:
                unconditional = predict(
                    inputs,
                    native_times[index],
                    negative,
                    negative_mask,
                    skipped=uncond_skip_layers if active else (),
                ).float()
                prediction = unconditional + guidance * (conditional - unconditional)
            if active and slg_layers and slg_scale > 0:
                skipped = predict(inputs, native_times[index], positive, positive_mask, slg_layers)
                if guidance == 1:
                    prediction = skipped
                prediction = prediction + slg_scale * (conditional - skipped)
            latents = latents + float(times[index + 1] - times[index]) * prediction.to(noise.dtype)
    return latents


@torch.no_grad()
def generate_latents(
    model: MossImageTransformer,
    positive: Caption,
    sampling: SamplingConfig,
    *,
    negative: Caption | None = None,
    vae_factor: int | None = None,
    condition: FluxCondition | None = None,
    references: tuple[Tensor, ...] = (),
    inpainting_mask: Tensor | None = None,
) -> Tensor:
    """Initialize native MOSS-Image noise and sample without changing model mode.

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
    validate_image_sampling(sampling, family="moss-image", variant=model.config.variant, version=None)
    if isinstance(positive, FluxTextFeatures):
        raise ValueError("Flux requires T5/CLIP features; other image models require Caption")
    if condition is not None:
        raise ValueError("Packed image conditions require Flux")
    if references and model.config.variant != "edit":
        raise ValueError("MOSS references require the edit variant")
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
    noise = torch.randn(
        (1, model.config.latent_channels, sampling.height // vae_factor, sampling.width // vae_factor),
        generator=generator,
    ).to(device=parameter.device, dtype=sampling_dtype(model))
    return sample_latents(
        model,
        noise,
        positive.features,
        positive_mask=positive.mask,
        negative=None if negative is None else negative.features,
        negative_mask=None if negative is None else negative.mask,
        steps=50 if sampling.steps is None else sampling.steps,
        shift=5.0 if sampling.shift is None else sampling.shift,
        training_shift=sampling.training_shift,
        guidance=3.0 if sampling.guidance is None else sampling.guidance,
        uncond_skip_layers=(() if model.config.variant == "edit" else (22,))
        if sampling.uncond_skip_layers is None
        else sampling.uncond_skip_layers,
        slg_layers=sampling.slg_layers,
        slg_scale=sampling.slg_scale,
        slg_start=sampling.slg_start,
        slg_stop=sampling.slg_stop,
        references=tuple(value.unsqueeze(0) if value.ndim == 3 else value for value in references),
    )
