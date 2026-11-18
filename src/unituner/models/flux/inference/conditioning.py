"""Encode native Flux inference image conditions with caller-local posterior draws."""

from pathlib import Path
from typing import Literal

import torch

from unituner.config.sampling import SamplingConfig, validate_image_sampling
from unituner.models.flux.conditioning import FluxCondition, image_ids, pack_fill_mask, pack_latents
from unituner.models.flux.data.media import read_fill_image, read_reference_image
from unituner.models.flux.vae import FluxVAE


@torch.no_grad()
def build_flux_condition(
    vae: FluxVAE, sampling: SamplingConfig, *, variant: Literal["dev", "kontext", "fill"]
) -> FluxCondition | None:
    """Return FP32 packed conditions on the borrowed VAE's current device.

    Kontext reference pixels retain independent native geometry; Fill uses the
    target grid and white-repaint masks. Sampling uses a separate CPU generator
    initialized with sampling.seed; the denoiser noise stream is independent.
    Mean mode consumes no noise. No global RNG, mode or residency changes occur.
    This explicit reproducible draw policy is not historical global-RNG parity.
    """
    validate_image_sampling(sampling, family="flux", variant=variant)
    if variant == "dev":
        return None
    if sampling.conditioning_image is None:
        raise ValueError("Flux image conditioning requires conditioning_image")
    device = next(vae.parameters()).device
    mask = None
    source = Path(sampling.conditioning_image)
    if variant == "fill":
        if vae.config.spatial_factor != 8:
            raise ValueError("Flux Fill masks require vae spatial_factor=8")
        pixels, mask = read_fill_image(
            source,
            None if sampling.mask_image is None else Path(sampling.mask_image),
            (sampling.width, sampling.height),
        )
    else:
        pixels = read_reference_image(source, preferred=sampling.conditioning_resize == "preferred")
    posterior = vae.posterior(pixels.unsqueeze(0).to(device))
    raw = posterior.mean
    if sampling.condition_posterior == "sample":
        generator = torch.Generator(device="cpu").manual_seed(sampling.seed)
        noise = torch.randn(raw.shape, generator=generator, dtype=torch.float32, device="cpu").to(device)
        raw = posterior.sample(noise)
    latents = vae.normalize(raw)
    tokens = pack_latents(latents)
    if mask is not None:
        return FluxCondition(tokens, fill_mask=pack_latents(pack_fill_mask(mask.unsqueeze(0).to(device))))
    return FluxCondition(
        tokens, image_ids(1, latents.shape[2] // 2, latents.shape[3] // 2, device=device, reference=True)
    )
