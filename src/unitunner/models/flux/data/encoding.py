"""Encode FLUX posterior draws with explicit caller-owned RNG and native normalization."""

import torch

from unitunner.models.flux.vae import FluxVAE


def encode_latents(vae: FluxVAE, pixels: torch.Tensor, generator: torch.Generator, *, sample: bool) -> torch.Tensor:
    """Encode one image with explicit CPU posterior draws and native normalization.

    A caller-owned generator fixes target-then-condition draw order. Return CHW
    model units without global RNG or module-mode mutation.
    """
    posterior = vae.posterior(pixels)
    raw = posterior.mean
    if sample:
        noise = torch.randn(raw.shape, generator=generator, device="cpu", dtype=torch.float32).to(raw.device)
        raw = posterior.sample(noise)
    return vae.normalize(raw)[0]
