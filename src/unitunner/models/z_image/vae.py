# Copyright (c) 2025 Z-Image Team. Apache-2.0.
# Adapted from https://github.com/Tongyi-MAI/Z-Image via Musubi Tuner.
# Original vae derived from https://github.com/black-forest-labs/flux.
# Modified to provide explicit posterior noise and typed native topology.
"""Encode/decode Z-Image pixels with native, Diffusers-layout VAE weights."""

from torch import Tensor, nn
from torch.nn import functional as F

from unitunner.artifacts.formats.precision import FloatingPrecision
from unitunner.models.components.posterior import DiagonalPosterior
from unitunner.models.z_image.config.vae import ZImageVAEConfig
from unitunner.precision import compute_context


class ResidualBlock(nn.Module):
    """Compute a two-convolution GroupNorm/SiLU residual with optional shortcut."""

    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def transform(self, x: Tensor) -> Tensor:
        """Return BCHW residual values using fused SiLU activations."""
        hidden = self.norm1(x)
        hidden = self.conv1(F.silu(hidden))
        hidden = self.norm2(hidden)
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return hidden + (x if self.conv_shortcut is None else self.conv_shortcut(x))

    forward = transform


class SpatialAttention(nn.Module):
    """Compute one-head spatial SDPA over H*W positions with a residual."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def attend(self, x: Tensor) -> Tensor:
        """Normalize BCHW pixels and return residual spatial attention in BCHW."""
        tokens = self.group_norm(x).flatten(2).transpose(1, 2)
        values = F.scaled_dot_product_attention(self.to_q(tokens), self.to_k(tokens), self.to_v(tokens), dropout_p=0.0)
        return x + self.to_out[0](values).transpose(1, 2).reshape_as(x)

    forward = attend


class MiddleBlock(nn.Module):
    """Compute residual-attention-residual with stable native artifact paths."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResidualBlock(channels, channels, groups), ResidualBlock(channels, channels, groups)]
        )
        self.attentions = nn.ModuleList([SpatialAttention(channels, groups)])

    def transform(self, x: Tensor) -> Tensor:
        """Return BCHW values without changing resolution or channel count."""
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))

    forward = transform


class Downsample(nn.Module):
    """Downsample asymmetrically padded pixels by a stride-two convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2)

    def downsample(self, x: Tensor) -> Tensor:
        """Convolve an already right/bottom padded BCHW tensor."""
        return self.conv(x)

    forward = downsample


class Upsample(nn.Module):
    """Upsample by nearest interpolation followed by a padded convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def upsample(self, x: Tensor) -> Tensor:
        """Return BCH(2)W(2) values with unchanged channels."""
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))

    forward = upsample


class EncoderBlock(nn.Module):
    """Own encoder residuals and optional native downsampler."""

    def __init__(self, in_channels: int, out_channels: int, config: ZImageVAEConfig, *, downsample: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResidualBlock(in_channels if index == 0 else out_channels, out_channels, config.norm_groups)
                for index in range(config.layers_per_block)
            ]
        )
        self.downsamplers = nn.ModuleList([Downsample(out_channels)]) if downsample else None

    def encode(self, x: Tensor) -> Tensor:
        """Compute residuals then apply native right/bottom zero padding if needed."""
        for block in self.resnets:
            x = block(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](F.pad(x, (0, 1, 0, 1)))
        return x

    forward = encode


class DecoderBlock(nn.Module):
    """Own decoder residuals and optional native upsampler."""

    def __init__(self, in_channels: int, out_channels: int, config: ZImageVAEConfig, *, upsample: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResidualBlock(in_channels if index == 0 else out_channels, out_channels, config.norm_groups)
                for index in range(config.layers_per_block + 1)
            ]
        )
        self.upsamplers = nn.ModuleList([Upsample(out_channels)]) if upsample else None

    def decode(self, x: Tensor) -> Tensor:
        """Compute residuals and optionally double spatial resolution."""
        for block in self.resnets:
            x = block(x)
        return x if self.upsamplers is None else self.upsamplers[0](x)

    forward = decode


class Encoder(nn.Module):
    """Map BCHW pixels to doubled latent-channel posterior parameters."""

    def __init__(self, config: ZImageVAEConfig) -> None:
        super().__init__()
        widths = config.block_channels
        self.conv_in = nn.Conv2d(config.in_channels, widths[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            [
                EncoderBlock(widths[max(index - 1, 0)], width, config, downsample=index < len(widths) - 1)
                for index, width in enumerate(widths)
            ]
        )
        self.mid_block = MiddleBlock(widths[-1], config.norm_groups)
        self.conv_norm_out = nn.GroupNorm(config.norm_groups, widths[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(widths[-1], config.latent_channels * 2, 3, padding=1)

    def encode(self, x: Tensor) -> Tensor:
        """Compute native posterior parameters, retaining encoder gradients."""
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        return self.conv_out(self.conv_act(self.conv_norm_out(self.mid_block(x))))

    forward = encode


class Decoder(nn.Module):
    """Map raw BCHW VAE latents to unconstrained native pixel values."""

    def __init__(self, config: ZImageVAEConfig) -> None:
        super().__init__()
        widths = tuple(reversed(config.block_channels))
        self.conv_in = nn.Conv2d(config.latent_channels, widths[0], 3, padding=1)
        self.mid_block = MiddleBlock(widths[0], config.norm_groups)
        self.up_blocks = nn.ModuleList(
            [
                DecoderBlock(widths[max(index - 1, 0)], width, config, upsample=index < len(widths) - 1)
                for index, width in enumerate(widths)
            ]
        )
        self.conv_norm_out = nn.GroupNorm(config.norm_groups, widths[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(widths[-1], config.out_channels, 3, padding=1)

    def decode(self, x: Tensor) -> Tensor:
        """Compute native pixel values without clipping or latent rescaling."""
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))

    forward = decode


class ZImageVAE(nn.Module):
    """Own a native VAE; expose raw posterior/decoder and explicit latent scaling.

    Pixel inputs normally use [-1,1]. Decoding does not clamp. Posterior mean or
    caller-supplied noise determines encoding; no hidden sampling or residency.
    Differentiation through the decoder remains available for restoration losses.
    """

    compute_precision: FloatingPrecision = "fp32"

    def __init__(self, config: ZImageVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quant_conv = (
            nn.Conv2d(2 * config.latent_channels, 2 * config.latent_channels, 1) if config.quant_conv else None
        )
        self.post_quant_conv = (
            nn.Conv2d(config.latent_channels, config.latent_channels, 1) if config.post_quant_conv else None
        )

    def encode(self, pixels: Tensor) -> DiagonalPosterior:
        """Encode BCHW pixels into raw mean/logvar, clamping logvar to [-30,20]."""
        factor = 2 ** (len(self.config.block_channels) - 1)
        if (
            pixels.ndim != 4
            or pixels.shape[1] != self.config.in_channels
            or min(pixels.shape[0], pixels.shape[2], pixels.shape[3]) <= 0
            or pixels.shape[2] % factor
            or pixels.shape[3] % factor
        ):
            raise ValueError("VAE pixels require BCHW shape with configured channels and divisible spatial sizes")
        with compute_context(pixels.device, self.compute_precision):
            parameters = self.encoder(pixels)
            if self.quant_conv is not None:
                parameters = self.quant_conv(parameters)
        mean, log_variance = parameters.float().chunk(2, dim=1)
        return DiagonalPosterior(mean, log_variance.clamp(-30, 20))

    def decode(self, latents: Tensor) -> Tensor:
        """Decode raw BCHW VAE latents without model normalization or clipping."""
        if latents.ndim != 4 or latents.shape[1] != self.config.latent_channels:
            raise ValueError("VAE latents require BCHW shape with configured channels")
        with compute_context(latents.device, self.compute_precision):
            return self.decoder(latents if self.post_quant_conv is None else self.post_quant_conv(latents))

    def normalize(self, latents: Tensor) -> Tensor:
        """Map raw VAE latents to model units: (latents-shift)*scale."""
        return (latents - self.config.shift_factor) * self.config.scaling_factor

    def denormalize(self, latents: Tensor) -> Tensor:
        """Map model latents to raw VAE units: latents/scale+shift."""
        return latents / self.config.scaling_factor + self.config.shift_factor
