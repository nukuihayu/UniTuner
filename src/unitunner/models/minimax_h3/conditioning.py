"""Encode frozen H3 visual condition posterior samples and deterministic stereo audio means."""

import torch
from torch import Tensor

from unitunner.models.minimax_h3.vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE


@torch.no_grad()
def encode_condition_video(vae: MiniMaxH3VideoVAE, pixels: Tensor) -> Tensor:
    """Encode seed42 condition samples with native FP16 roundtrip before normalization."""
    generator = torch.Generator(device="cpu").manual_seed(42)
    posterior = vae.posterior(pixels)
    noise = torch.randn(posterior.mean.shape, generator=generator, dtype=torch.float32, device="cpu").to(
        posterior.mean.device
    )
    return vae.normalize(posterior.sample(noise).to(torch.float16).to(torch.float32))


@torch.no_grad()
def encode_audio(vae: MiniMaxH3AudioVAE, waveform: Tensor) -> Tensor:
    """Move32kHz stereo2L audio to the VAE and return native B32ST means.

    Caller owns waveform decoding/resampling and peak scaling. The model's
    current parameter device is authoritative; condition audio always uses FP32.
    Local views do not modify the waveform, weights, mode or global RNG.
    """
    if waveform.ndim != 2 or waveform.shape[0] != 2 or waveform.shape[1] < 1 or not waveform.is_floating_point():
        raise ValueError("H3 condition audio requires a nonempty floating stereo [2,L] waveform at32kHz")
    device = next(vae.parameters()).device
    return vae.encode(waveform.unsqueeze(0).to(device=device, dtype=torch.float32))
