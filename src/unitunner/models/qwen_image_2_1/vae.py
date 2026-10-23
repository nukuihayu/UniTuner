"""Compute and load native qwen image 2 1 VAEs with explicit execution ownership."""

from itertools import pairwise
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unitunner.artifacts.formats.precision import FloatingPrecision
from unitunner.models.components.posterior import DiagonalPosterior
from unitunner.models.qwen_image.vae import (
    QwenImageVAEExecution,
    QwenImageVAETiling,
    VAEAttention,
    VAERMSNorm,
    execute_vae,
)
from unitunner.models.qwen_image_2_1.config.vae import QwenImage21VAEConfig
from unitunner.precision import compute_context

# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.


class ImageConv2d(nn.Conv2d):
    """Apply native spatial Conv2d to BC1HW images with no temporal history."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, padding: int = 0) -> None:
        super().__init__(in_channels, out_channels, kernel_size)
        self._spatial_padding = (padding, padding, padding, padding)

    def convolve(self, input: Tensor) -> Tensor:
        """Return BC1HW pixels; reject video before the spatial convolution."""
        values = input
        if values.ndim != 5 or values.shape[2] != 1:
            raise ValueError("Qwen Image2.1 convolution requires BC1HW images without temporal history")
        return super().forward(F.pad(values.squeeze(2), self._spatial_padding)).unsqueeze(2)

    forward = convolve


class AverageDownsample(nn.Module):
    """Fold spatial/temporal axes into channels and average native channel groups."""

    def __init__(self, in_channels: int, out_channels: int, *, temporal_factor: int, spatial_factor: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.temporal_factor = temporal_factor
        self.spatial_factor = spatial_factor
        self.group_size = in_channels * temporal_factor * spatial_factor**2 // out_channels

    def downsample(self, values: Tensor) -> Tensor:
        """Return the native shortcut; front-pad singleton time before averaging."""
        temporal, spatial = self.temporal_factor, self.spatial_factor
        values = F.pad(values, (0, 0, 0, 0, (temporal - values.shape[2] % temporal) % temporal, 0))
        batch, channels, frames, height, width = values.shape
        folded = values.reshape(
            batch, channels, frames // temporal, temporal, height // spatial, spatial, width // spatial, spatial
        )
        folded = folded.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        return folded.reshape(
            batch, self.out_channels, self.group_size, frames // temporal, height // spatial, width // spatial
        ).mean(dim=2)

    forward = downsample


class DuplicateUpsample(nn.Module):
    """Repeat channels then unfold native temporal/spatial axes for the shortcut."""

    def __init__(self, in_channels: int, out_channels: int, *, temporal_factor: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.temporal_factor = temporal_factor
        self.repeats = out_channels * temporal_factor * 4 // in_channels

    def upsample(self, values: Tensor) -> Tensor:
        """Return BC1(2H)(2W) with first-image temporal cropping and gradients."""
        batch, _, frames, height, width = values.shape
        values = values.repeat_interleave(self.repeats, dim=1)
        values = values.reshape(batch, self.out_channels, self.temporal_factor, 2, 2, frames, height, width)
        values = values.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        values = values.reshape(batch, self.out_channels, frames * self.temporal_factor, height * 2, width * 2)
        return values[:, :, self.temporal_factor - 1 :]

    forward = upsample


class ImageResample(nn.Module):
    """Own spatial resampling and native registered but unused temporal weights."""

    def __init__(
        self, channels: int, *, mode: Literal["down", "up"], temporal: bool, up_channels: int | None = None
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "down":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(channels, channels, 3, stride=2))
        else:
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="nearest-exact"),
                nn.Conv2d(channels, channels // 2 if up_channels is None else up_channels, 3, padding=1),
            )
        self.time_conv: ImageConv2d | None = (
            ImageConv2d(channels, channels * 2 if mode == "up" else channels, 1) if temporal else None
        )

    def resample_image(self, values: Tensor) -> Tensor:
        """Apply spatial resampling only, preserving singleton temporal geometry."""
        return self.resample(values.squeeze(2)).unsqueeze(2)

    forward = resample_image


def fold_patches(values: Tensor, patch_size: int) -> Tensor:
    """Fold BC1HW spatial patch width before height into channels, preserving gradients."""
    if patch_size == 1:
        return values
    batch, channels, frames, height, width = values.shape
    return (
        values.reshape(batch, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size)
        .permute(0, 1, 6, 4, 2, 3, 5)
        .reshape(batch, channels * patch_size**2, frames, height // patch_size, width // patch_size)
    )


def unfold_patches(values: Tensor, patch_size: int) -> Tensor:
    """Invert width-before-height channel folding without casting or detaching."""
    if patch_size == 1:
        return values
    batch, channels, frames, height, width = values.shape
    return (
        values.reshape(batch, channels // patch_size**2, patch_size, patch_size, frames, height, width)
        .permute(0, 1, 4, 5, 3, 6, 2)
        .reshape(batch, channels // patch_size**2, frames, height * patch_size, width * patch_size)
    )


# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.


class ImageResidual(nn.Module):
    """Compute two normalized image convolutions with an optional channel shortcut."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = VAERMSNorm(in_channels, temporal=True)
        self.conv1 = ImageConv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = VAERMSNorm(out_channels, temporal=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = ImageConv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = ImageConv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def transform(self, values: Tensor) -> Tensor:
        """Return BC1HW residual features, retaining all image/parameter gradients."""
        shortcut = self.conv_shortcut(values)
        values = self.conv1(F.silu(self.norm1(values)))
        return self.conv2(self.dropout(F.silu(self.norm2(values)))) + shortcut

    forward = transform


class ImageMiddle(nn.Module):
    """Preserve native residual, spatial attention, residual weight names and order."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [ImageResidual(channels, channels, dropout), ImageResidual(channels, channels, dropout)]
        )
        self.attentions = nn.ModuleList([VAEAttention(channels)])

    def transform(self, values: Tensor) -> Tensor:
        """Attend only within a single image, without temporal histories."""
        return self.resnets[1](self.attentions[0](self.resnets[0](values)))

    forward = transform


class ResidualDownsample(nn.Module):
    """Add folded channel averaging to native residual/spatial downsampling."""

    def __init__(
        self, in_channels: int, out_channels: int, config: QwenImage21VAEConfig, *, transition: bool, temporal: bool
    ) -> None:
        super().__init__()
        self.avg_shortcut = AverageDownsample(
            in_channels, out_channels, temporal_factor=2 if temporal else 1, spatial_factor=2 if transition else 1
        )
        self.resnets = nn.ModuleList(
            [
                ImageResidual(in_channels if index == 0 else out_channels, out_channels, config.dropout)
                for index in range(config.layers_per_block)
            ]
        )
        self.downsampler = ImageResample(out_channels, mode="down", temporal=temporal) if transition else None

    def downsample(self, values: Tensor) -> Tensor:
        """Return downsampled BC1HW with native shortcut zero-padding and averaging."""
        shortcut = self.avg_shortcut(values)
        for block in self.resnets:
            values = block(values)
        if self.downsampler is not None:
            values = self.downsampler(values)
        return values + shortcut

    forward = downsample


class ImageUpBlock(nn.Module):
    """Compute native extra decoder residuals and optional half-width upsampling."""

    def __init__(
        self, in_channels: int, out_channels: int, config: QwenImage21VAEConfig, *, transition: bool, temporal: bool
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ImageResidual(in_channels if index == 0 else out_channels, out_channels, config.dropout)
                for index in range(config.layers_per_block + 1)
            ]
        )
        self.upsamplers = (
            nn.ModuleList([ImageResample(out_channels, mode="up", temporal=temporal)]) if transition else None
        )

    def transform(self, values: Tensor) -> Tensor:
        """Return inverse spatial features with no image-level resampling shortcut."""
        for block in self.resnets:
            values = block(values)
        return self.upsamplers[0](values) if self.upsamplers is not None else values

    forward = transform


class ResidualUpsample(nn.Module):
    """Add repeated/unfolded channel shortcut to native full-width upsampling."""

    def __init__(
        self, in_channels: int, out_channels: int, config: QwenImage21VAEConfig, *, transition: bool, temporal: bool
    ) -> None:
        super().__init__()
        self.avg_shortcut = (
            DuplicateUpsample(in_channels, out_channels, temporal_factor=2 if temporal else 1) if transition else None
        )
        self.resnets = nn.ModuleList(
            [
                ImageResidual(in_channels if index == 0 else out_channels, out_channels, config.dropout)
                for index in range(config.layers_per_block + 1)
            ]
        )
        self.upsampler = (
            ImageResample(out_channels, mode="up", temporal=temporal, up_channels=out_channels) if transition else None
        )

    def upsample(self, values: Tensor) -> Tensor:
        """Return inverse spatial features, cropping the shortcut to the first image."""
        shortcut = self.avg_shortcut(values) if self.avg_shortcut is not None else None
        for block in self.resnets:
            values = block(values)
        if self.upsampler is not None:
            values = self.upsampler(values)
        return values + shortcut if shortcut is not None else values

    forward = upsample


class ImageEncoder(nn.Module):
    """Own native asymmetric-width image encoder and raw Gaussian moment output."""

    def __init__(self, config: QwenImage21VAEConfig) -> None:
        super().__init__()
        widths = [config.base_channels * value for value in (1, *config.channel_multipliers)]
        self.conv_in = ImageConv2d(config.input_channels * config.patch_size**2, widths[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        for index, (in_channels, out_channels) in enumerate(pairwise(widths)):
            transition = index < len(config.channel_multipliers) - 1
            temporal = transition and config.temporal_downsample[index]
            if config.is_residual:
                self.down_blocks.append(
                    ResidualDownsample(in_channels, out_channels, config, transition=transition, temporal=temporal)
                )
            else:
                for _ in range(config.layers_per_block):
                    self.down_blocks.append(ImageResidual(in_channels, out_channels, config.dropout))
                    if 2.0**-index in config.attention_scales:
                        self.down_blocks.append(VAEAttention(out_channels))
                    in_channels = out_channels
                if transition:
                    self.down_blocks.append(ImageResample(out_channels, mode="down", temporal=temporal))
        self.mid_block = ImageMiddle(widths[-1], config.dropout)
        self.norm_out = VAERMSNorm(widths[-1], temporal=True)
        self.conv_out = ImageConv2d(widths[-1], 2 * config.latent_channels, 3, padding=1)

    def encode(self, values: Tensor) -> Tensor:
        """Return unnormalized Gaussian moments in BC1HW at the latent resolution."""
        values = self.conv_in(values)
        for block in self.down_blocks:
            values = block(values)
        return self.conv_out(F.silu(self.norm_out(self.mid_block(values))))

    forward = encode


class ImageDecoder(nn.Module):
    """Own native independent-width image decoder and packed pixel output."""

    def __init__(self, config: QwenImage21VAEConfig) -> None:
        super().__init__()
        widths = [
            config.decoder_base_channels * value
            for value in (config.channel_multipliers[-1], *config.channel_multipliers[::-1])
        ]
        self.conv_in = ImageConv2d(config.latent_channels, widths[0], 3, padding=1)
        self.mid_block = ImageMiddle(widths[0], config.dropout)
        self.up_blocks = nn.ModuleList()
        for index, (in_channels, out_channels) in enumerate(pairwise(widths)):
            transition = index < len(config.channel_multipliers) - 1
            temporal = transition and config.temporal_downsample[::-1][index]
            if config.is_residual:
                self.up_blocks.append(
                    ResidualUpsample(in_channels, out_channels, config, transition=transition, temporal=temporal)
                )
            else:
                self.up_blocks.append(
                    ImageUpBlock(
                        in_channels // 2 if index else in_channels,
                        out_channels,
                        config,
                        transition=transition,
                        temporal=temporal,
                    )
                )
        self.norm_out = VAERMSNorm(widths[-1], temporal=True)
        self.conv_out = ImageConv2d(widths[-1], config.input_channels * config.patch_size**2, 3, padding=1)

    def decode(self, values: Tensor) -> Tensor:
        """Return unclamped packed BC1HW pixels without cross-call state."""
        values = self.mid_block(self.conv_in(values))
        for block in self.up_blocks:
            values = block(values)
        return self.conv_out(F.silu(self.norm_out(values)))

    forward = decode


# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.


class QwenImage21VAE(nn.Module):
    """Own image-only native FP32 VAE weights, with no retained histories or policy.

    BC1HW pixels preserve RGB/RGBA channels; temporal flags affect registered
    native weights and shortcut scaling, never authorize video. Default encoding
    uses means, as the pinned Musubi cache caller does. Sampling requires explicit
    noise. Calls preserve caller mode/residency; RNG changes only when configured
    nonzero dropout is active in training mode. All paths retain gradients.
    """

    compute_precision: FloatingPrecision = "fp32"

    def __init__(self, config: QwenImage21VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ImageEncoder(config)
        self.quant_conv = ImageConv2d(2 * config.latent_channels, 2 * config.latent_channels, 1)
        self.post_quant_conv = ImageConv2d(config.latent_channels, config.latent_channels, 1)
        self.decoder = ImageDecoder(config)

    def _validate(self, values: Tensor, *, pixels: bool) -> None:
        """Validate singleton-frame FP32 geometry before convolution or tiling."""
        parameter = next(self.parameters())
        channels = self.config.input_channels if pixels else self.config.latent_channels
        if (
            values.ndim != 5
            or min(values.shape) < 1
            or values.shape[2] != 1
            or values.shape[1] != channels
            or values.dtype != torch.float32
            or parameter.dtype != torch.float32
            or values.device != parameter.device
        ):
            raise ValueError("Qwen Image2.1 vae requires nonempty FP32 BC1HW with matching channels/device")
        if pixels and any(size % self.config.spatial_factor for size in values.shape[-2:]):
            raise ValueError("Qwen Image2.1 pixel H/W must be divisible by spatial_factor")

    def _normalization(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Build native FP32 per-channel mean/std directly on the latent device."""
        shape = (1, self.config.latent_channels, 1, 1, 1)
        return values.new_tensor(self.config.latents_mean).reshape(shape), values.new_tensor(
            self.config.latents_std
        ).reshape(shape)

    def normalize(self, latents: Tensor) -> Tensor:
        """Apply native (raw-mean)/std to BC1HW latents; do not substitute reciprocal multiplication."""
        self._validate(latents, pixels=False)
        mean, std = self._normalization(latents)
        return (latents - mean) / std

    def denormalize(self, latents: Tensor) -> Tensor:
        """Apply native normalized*std+mean to BC1HW latents without sampling."""
        self._validate(latents, pixels=False)
        mean, std = self._normalization(latents)
        return latents * std + mean

    def _execution(self, execution: QwenImageVAEExecution | None) -> QwenImageVAEExecution | None:
        """Translate pixel tile geometry to packed-pixel geometry before any convolution."""
        if execution is None or execution.tiling is None:
            return execution
        tile = execution.tiling
        values = (tile.height, tile.width, tile.stride_height, tile.stride_width)
        if any(value % self.config.spatial_factor for value in values):
            raise ValueError("Qwen Image2.1 tile sizes/strides must be multiples of spatial_factor")
        patch = self.config.patch_size
        return QwenImageVAEExecution(
            batch_slice_size=execution.batch_slice_size,
            tiling=QwenImageVAETiling(
                height=tile.height // patch,
                width=tile.width // patch,
                stride_height=tile.stride_height // patch,
                stride_width=tile.stride_width // patch,
            ),
        )

    def _encode_moments(self, pixels: Tensor) -> Tensor:
        """Return raw Gaussian moments for one packed-pixel batch slice/tile."""
        return self.quant_conv(self.encoder(pixels))

    def posterior(self, pixels: Tensor, *, execution: QwenImageVAEExecution | None = None) -> DiagonalPosterior:
        """Return raw BC1HW mean/log-variance, clamping log-variance to [-30,20].

        Pixels use [-1,1]. Optional spatial tiles use native overlap/crop ordering
        before moment splitting/clamping. Batch slices are independent. No
        posterior sampling or latent normalization occurs at this boundary.
        """
        self._validate(pixels, pixels=True)
        policy = self._execution(execution)
        with compute_context(pixels.device, self.compute_precision):
            moments = execute_vae(
                fold_patches(pixels, self.config.patch_size),
                self._encode_moments,
                execution=policy,
                spatial_factor=self.config.spatial_factor // self.config.patch_size,
                encoding=True,
            )
            mean, log_variance = moments.float().chunk(2, dim=1)
            return DiagonalPosterior(mean, log_variance.clamp(-30, 20))

    def encode(
        self, pixels: Tensor, *, posterior_noise: Tensor | None = None, execution: QwenImageVAEExecution | None = None
    ) -> Tensor:
        """Encode BC1HW pixels into normalized latents, selecting mean or explicit-noise sample."""
        posterior = self.posterior(pixels, execution=execution)
        return self.normalize(posterior.mean if posterior_noise is None else posterior.sample(posterior_noise))

    def _decode_pixels(self, latents: Tensor) -> Tensor:
        """Decode a raw latent tile into packed pixels without clipping."""
        return self.decoder(self.post_quant_conv(latents))

    def decode_raw(self, latents: Tensor, *, execution: QwenImageVAEExecution | None = None) -> Tensor:
        """Decode raw BC1HW latents and clamp [-1,1] after tile blending/patch unfolding.

        Spatial tiles approximate full-image attention/convolution; unlike the
        previous Qwen VAE, both whole and tiled Qwen2.1 output is clamped.
        Registered time convolutions are intentionally inactive for images.
        """
        self._validate(latents, pixels=False)
        policy = self._execution(execution)
        with compute_context(latents.device, self.compute_precision):
            packed = execute_vae(
                latents,
                self._decode_pixels,
                execution=policy,
                spatial_factor=self.config.spatial_factor // self.config.patch_size,
                encoding=False,
            )
            return unfold_patches(packed, self.config.patch_size).clamp(-1, 1)

    def decode(self, latents: Tensor, *, execution: QwenImageVAEExecution | None = None) -> Tensor:
        """Decode normalized BC1HW latents to same-channel pixels with optional BF16 compute."""
        return self.decode_raw(self.denormalize(latents), execution=execution)
