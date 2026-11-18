# Adapted from Black Forest Labs FLUX, Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""Encode FLUX.2/MOSS posterior means with packed checkpoint running-stat normalization."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unituner.artifacts.formats.precision import FloatingPrecision
from unituner.models.components.autoencoder import AutoencoderDecoder, AutoencoderEncoder
from unituner.models.flux2.config.vae import Flux2VAEConfig
from unituner.precision import compute_context


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
        self.encoder = AutoencoderEncoder(config, quant_conv=True)
        self.decoder = AutoencoderDecoder(config, post_quant_conv=True)
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
