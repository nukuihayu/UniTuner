# Adapted from Black Forest Labs FLUX, Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""Compute native FLUX convolutional residual, attention and resampling blocks."""

from torch import Tensor, nn
from torch.nn import functional as F


def swish(value: Tensor) -> Tensor:
    """Compute x*sigmoid(x) with fused SiLU and one saved activation for backward."""
    return F.silu(value)


class SpatialAttention(nn.Module):
    """Compute one-head spatial SDPA with native convolutional QKV/output keys."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.q, self.k, self.v = (nn.Conv2d(channels, channels, 1) for _ in range(3))
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def attend(self, pixels: Tensor) -> Tensor:
        """Attend over BCHW spatial positions, preserving the input residual graph."""
        normalized = self.norm(pixels)
        query, key, value = [
            projection(normalized).flatten(2).transpose(1, 2).unsqueeze(1).contiguous()
            for projection in (self.q, self.k, self.v)
        ]
        output = F.scaled_dot_product_attention(query, key, value).squeeze(1).transpose(1, 2).reshape_as(pixels)
        return pixels + self.proj_out(output)

    forward = attend


class ResidualConvBlock(nn.Module):
    """Compute two normalized convolutions and an optional native channel shortcut."""

    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def transform(self, pixels: Tensor) -> Tensor:
        """Return BCHW residual features using x*sigmoid(x) activation."""
        hidden = self.conv1(swish(self.norm1(pixels)))
        hidden = self.conv2(swish(self.norm2(hidden)))
        residual = pixels if self.nin_shortcut is None else self.nin_shortcut(pixels)
        return residual + hidden

    forward = transform


class Downsample2d(nn.Module):
    """Halve spatial dimensions after asymmetric right/bottom zero padding."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2)

    def downsample(self, pixels: Tensor) -> Tensor:
        """Return BCH/2W/2 features for an even-resolution input."""
        return self.conv(F.pad(pixels, (0, 1, 0, 1)))

    forward = downsample


class Upsample2d(nn.Module):
    """Double spatial dimensions with nearest interpolation and convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def upsample(self, pixels: Tensor) -> Tensor:
        """Return BC(2H)(2W) features without changing channel count."""
        return self.conv(F.interpolate(pixels, scale_factor=2.0, mode="nearest"))

    forward = upsample


class AutoencoderMiddle(nn.Module):
    """Own native residual-attention-residual bottleneck keys."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.block_1 = ResidualConvBlock(channels, channels, groups)
        self.attn_1 = SpatialAttention(channels, groups)
        self.block_2 = ResidualConvBlock(channels, channels, groups)

    def transform(self, pixels: Tensor) -> Tensor:
        """Return BCHW bottleneck features without spatial resampling."""
        return self.block_2(self.attn_1(self.block_1(pixels)))

    forward = transform


class AutoencoderStage(nn.Module):
    """Own already-constructed residual blocks and an optional native resampler."""

    def __init__(
        self, blocks: nn.ModuleList, *, downsample: Downsample2d | None = None, upsample: Upsample2d | None = None
    ) -> None:
        super().__init__()
        self.block = blocks
        self.downsample = downsample
        self.upsample = upsample

    def transform(self, pixels: Tensor) -> Tensor:
        """Apply ordered residuals then the selected spatial resampler."""
        for block in self.block:
            pixels = block(pixels)
        if self.downsample is not None:
            pixels = self.downsample(pixels)
        if self.upsample is not None:
            pixels = self.upsample(pixels)
        return pixels

    forward = transform
