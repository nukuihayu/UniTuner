# Adapted from Black Forest Labs FLUX, Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""Encode native FLUX.1 posterior moments and explicitly normalized image latents."""

import torch
from torch import Tensor, nn

from unituner.artifacts.formats.precision import FloatingPrecision
from unituner.models.components.autoencoder import AutoencoderDecoder, AutoencoderEncoder
from unituner.models.components.posterior import DiagonalPosterior
from unituner.models.flux.config.vae import FluxVAEConfig
from unituner.precision import compute_context


class FluxVAE(nn.Module):
    """Own native image encoding/decoding, FP32 weights and optional BF16 compute.

    Pixels use BCHW [-1,1]. Encode returns (raw-shift)*scale model latents;
    decode inverts normalization, without clipping. Mean encoding is explicit
    by omission of noise; sampled encoding requires caller-supplied posterior
    noise. Native FLUX log-variance is unclamped, unlike the Z-Image VAE.
    No hidden RNG, mode changes or residency transfers occur. Both paths retain
    gradients, including through frozen weights to pixels/latents.
    """

    compute_precision: FloatingPrecision = "fp32"

    def __init__(self, config: FluxVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AutoencoderEncoder(config)
        self.decoder = AutoencoderDecoder(config)

    def _validate(self, values: Tensor, channels: int) -> None:
        if values.ndim != 4 or min(values.shape) < 1 or values.shape[1] != channels or values.dtype != torch.float32:
            raise ValueError("FLUX vae requires nonempty FP32 BCHW with configured channels")
        if any(
            parameter.dtype != torch.float32 or parameter.device != values.device for parameter in self.parameters()
        ):
            raise ValueError("FLUX vae weights must be FP32 on the input device")

    def posterior(self, pixels: Tensor) -> DiagonalPosterior:
        """Return native unclamped Gaussian BCHW moments; suppress ambient autocast."""
        self._validate(pixels, self.config.in_channels)
        if any(size % self.config.spatial_factor for size in pixels.shape[-2:]):
            raise ValueError("FLUX vae pixels must be divisible by spatial_factor")
        with compute_context(pixels.device, self.compute_precision):
            mean, log_variance = self.encoder(pixels).float().chunk(2, dim=1)
        return DiagonalPosterior(mean, log_variance)

    def encode(self, pixels: Tensor, *, noise: Tensor | None = None) -> Tensor:
        """Return normalized posterior mean or explicitly sampled FP32 BCHW latents."""
        posterior = self.posterior(pixels)
        raw = posterior.mean if noise is None else posterior.sample(noise)
        return self.normalize(raw)

    def decode(self, latents: Tensor) -> Tensor:
        """Decode normalized FP32 BCHW latents under the configured VAE compute precision."""
        self._validate(latents, self.config.latent_channels)
        with compute_context(latents.device, self.compute_precision):
            return self.decoder(self.denormalize(latents))

    def normalize(self, latents: Tensor) -> Tensor:
        """Return (raw-shift)*scale model units without packing or mutation."""
        return (latents - self.config.shift_factor) * self.config.scaling_factor

    def denormalize(self, latents: Tensor) -> Tensor:
        """Return model/scale+shift raw VAE units without mutation."""
        return latents / self.config.scaling_factor + self.config.shift_factor
