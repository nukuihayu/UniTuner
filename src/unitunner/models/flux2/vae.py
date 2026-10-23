# Adapted from Black Forest Labs FLUX, Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""Encode FLUX.2/MOSS posterior means with packed checkpoint running-stat normalization."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unitunner.artifacts.formats.precision import FloatingPrecision
from unitunner.models.components.autoencoder import (
    AutoencoderMiddle,
    AutoencoderStage,
    Downsample2d,
    ResidualConvBlock,
    Upsample2d,
    swish,
)
from unitunner.models.flux2.config.vae import Flux2VAEConfig
from unitunner.precision import compute_context


class Encoder(nn.Module):
    """Own native posterior-moment convolutions, including encoder quant_conv."""

    def __init__(self, config: Flux2VAEConfig) -> None:
        super().__init__()
        self.quant_conv = nn.Conv2d(2 * config.latent_channels, 2 * config.latent_channels, 1)
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
        """Return BCHW native posterior moments, preserving the pixel gradient graph."""
        hidden = self.conv_in(pixels)
        for stage in self.down:
            hidden = stage(hidden)
        return self.quant_conv(self.conv_out(swish(self.norm_out(self.mid(hidden)))))

    forward = encode


class Decoder(nn.Module):
    """Own native inverse-posterior convolutions and ascending-resolution stages."""

    def __init__(self, config: Flux2VAEConfig) -> None:
        super().__init__()
        self.post_quant_conv = nn.Conv2d(config.latent_channels, config.latent_channels, 1)
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
        """Return unconstrained native BCHW pixels without clipping or scaling."""
        hidden = self.mid(self.conv_in(self.post_quant_conv(latents)))
        hidden = hidden.to(next(self.up.parameters()).dtype)
        for stage in reversed(self.up):
            hidden = stage(hidden)
        return self.conv_out(swish(self.norm_out(hidden)))

    forward = decode


class Flux2VAE(nn.Module):
    """Own deterministic FLUX.2 encoding/decoding with optional BF16 compute.

    Images use BCHW [-1,1]. Encode uses posterior mean only, packs 2x2 spatial
    cells into channels in (channel,row_offset,column_offset) order, then uses
    checkpoint BN running statistics. Decode inverts both operations. Statistics
    never update, including in train mode; no mode/RNG/residency mutation occurs.
    Encoder and decoder remain differentiable. The caller owns parameter freezing.
    """

    compute_precision: FloatingPrecision = "fp32"

    def __init__(self, config: Flux2VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.bn = nn.BatchNorm2d(
            4 * config.latent_channels,
            eps=config.normalization_eps,
            momentum=0.1,
            affine=False,
            track_running_stats=True,
        )

    def _validate(self, value: Tensor, channels: int) -> None:
        if (
            value.ndim != 4
            or not value.numel()
            or value.shape[1] != channels
            or not value.is_floating_point()
            or value.device != self.encoder.conv_in.weight.device
        ):
            raise ValueError("FLUX.2 vae requires nonempty floating BCHW on its model device")
        if (
            self.encoder.conv_in.weight.dtype != torch.float32
            or self.bn.running_mean is None
            or self.bn.running_var is None
            or self.bn.running_mean.dtype != torch.float32
            or self.bn.running_var.dtype != torch.float32
        ):
            raise ValueError("FLUX.2 vae requires FP32 weights and running statistics")

    def normalize(self, latents: Tensor) -> Tensor:
        """Apply frozen BN statistics to packed BCHW latents without changing modes."""
        self._validate(latents, 4 * self.config.latent_channels)
        with torch.autocast(device_type=latents.device.type, enabled=False):
            return F.batch_norm(
                latents.float(),
                self.bn.running_mean,
                self.bn.running_var,
                training=False,
                eps=self.config.normalization_eps,
            )

    def denormalize(self, latents: Tensor) -> Tensor:
        """Invert packed latent normalization using FP32 running mean/variance."""
        self._validate(latents, 4 * self.config.latent_channels)
        variance, mean = self.bn.running_var, self.bn.running_mean
        if variance is None or mean is None:
            raise RuntimeError("FLUX.2 vae normalization statistics are missing")
        with torch.autocast(device_type=latents.device.type, enabled=False):
            return latents.float() * (variance.view(1, -1, 1, 1) + self.config.normalization_eps).sqrt() + mean.view(
                1, -1, 1, 1
            )

    def encode(self, pixels: Tensor) -> Tensor:
        """Return FP32 normalized packed BCHW posterior means without drawing noise.

        Spatial dimensions must be divisible by config.spatial_factor. FP32
        parameters/statistics are required; ambient autocast is disabled. Convert
        floating pixels to FP32 while retaining gradients to the original tensor.
        """
        self._validate(pixels, self.config.in_channels)
        if pixels.shape[2] % self.config.spatial_factor or pixels.shape[3] % self.config.spatial_factor:
            raise ValueError("FLUX.2 vae pixel dimensions must be divisible by spatial_factor")
        with compute_context(pixels.device, self.compute_precision):
            mean = self.encoder(pixels.float()).float().chunk(2, dim=1)[0]
            batch, channels, height, width = mean.shape
            packed = (
                mean.reshape(batch, channels, height // 2, 2, width // 2, 2)
                .permute(0, 1, 3, 5, 2, 4)
                .reshape(batch, 4 * channels, height // 2, width // 2)
            )
            return self.normalize(packed)

    def decode(self, latents: Tensor) -> Tensor:
        """Invert packing/statistics and decode unclipped BCHW pixels with optional BF16 compute."""
        self._validate(latents, 4 * self.config.latent_channels)
        with compute_context(latents.device, self.compute_precision):
            raw = self.denormalize(latents)
            batch, _, height, width = raw.shape
            unpacked = (
                raw.reshape(batch, self.config.latent_channels, 2, 2, height, width)
                .permute(0, 1, 4, 2, 5, 3)
                .reshape(batch, self.config.latent_channels, height * 2, width * 2)
            )
            return self.decoder(unpacked)
