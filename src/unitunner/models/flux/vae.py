# Adapted from Black Forest Labs FLUX, Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""Encode native FLUX.1 posterior moments and explicitly normalized image latents."""

import torch
from torch import Tensor, nn

from unitunner.artifacts.formats.precision import FloatingPrecision
from unitunner.models.components.autoencoder import (
    AutoencoderMiddle,
    AutoencoderStage,
    Downsample2d,
    ResidualConvBlock,
    Upsample2d,
    swish,
)
from unitunner.models.components.posterior import DiagonalPosterior
from unitunner.models.flux.config.vae import FluxVAEConfig
from unitunner.precision import compute_context


class Encoder(nn.Module):
    """Own native convolutional encoder keys without quant-convolution parameters."""

    def __init__(self, config: FluxVAEConfig) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(config.in_channels, config.base_channels, 3, padding=1)
        self.down = nn.ModuleList()
        width = config.base_channels
        for index, multiplier in enumerate(config.channel_multipliers):
            output_width = config.base_channels * multiplier
            blocks = nn.ModuleList()
            for _ in range(config.layers_per_block):
                blocks.append(ResidualConvBlock(width, output_width, config.norm_groups))
                width = output_width
            self.down.append(
                AutoencoderStage(
                    blocks, downsample=Downsample2d(width) if index < len(config.channel_multipliers) - 1 else None
                )
            )
        self.mid = AutoencoderMiddle(width, config.norm_groups)
        self.norm_out = nn.GroupNorm(config.norm_groups, width, eps=1e-6)
        self.conv_out = nn.Conv2d(width, 2 * config.latent_channels, 3, padding=1)

    def encode(self, pixels: Tensor) -> Tensor:
        """Return native BCHW mean/log-variance moments with a complete gradient path."""
        hidden = self.conv_in(pixels)
        for stage in self.down:
            hidden = stage(hidden)
        return self.conv_out(swish(self.norm_out(self.mid(hidden))))

    forward = encode


class Decoder(nn.Module):
    """Own native decoder stages indexed in checkpoint resolution order."""

    def __init__(self, config: FluxVAEConfig) -> None:
        super().__init__()
        width = config.base_channels * config.channel_multipliers[-1]
        self.conv_in = nn.Conv2d(config.latent_channels, width, 3, padding=1)
        self.mid = AutoencoderMiddle(width, config.norm_groups)
        self.up = nn.ModuleList()
        for index in reversed(range(len(config.channel_multipliers))):
            output_width = config.base_channels * config.channel_multipliers[index]
            blocks = nn.ModuleList()
            for _ in range(config.layers_per_block + 1):
                blocks.append(ResidualConvBlock(width, output_width, config.norm_groups))
                width = output_width
            self.up.insert(0, AutoencoderStage(blocks, upsample=Upsample2d(width) if index else None))
        self.norm_out = nn.GroupNorm(config.norm_groups, width, eps=1e-6)
        self.conv_out = nn.Conv2d(width, config.out_channels, 3, padding=1)

    def decode(self, latents: Tensor) -> Tensor:
        """Return BCHW unclamped pixels from raw VAE latents."""
        hidden = self.mid(self.conv_in(latents))
        for stage in reversed(self.up):
            hidden = stage(hidden)
        return self.conv_out(swish(self.norm_out(hidden)))

    forward = decode


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
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

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
