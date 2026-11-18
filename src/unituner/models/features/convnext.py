# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from facebookresearch/dinov3; see licenses/DINOv3-LICENSE.md.
"""Native, frozen DINOv3 feature backbones; no Hub or Transformers dependency.

Retain the full official ConvNeXt state contract. Restoration uses the first
three stages with optional L2 pooling inside the hierarchy.
"""

from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ChannelLayerNorm(nn.LayerNorm):
    """Official channel-first normalization with FP32 statistics for FP32 input."""

    def __init__(self, width: int) -> None:
        super().__init__(width, eps=1e-6)

    def compute(self, input: torch.Tensor) -> torch.Tensor:
        # Native LayerNorm accumulates reduced-precision statistics in FP32 and
        # avoids materializing channel-sized centered/squared intermediates.
        return F.layer_norm(
            input.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)

    forward = compute


class ConvNeXtBlock(nn.Module):
    """Apply depthwise 7x7 convolution, channels-last MLP and learned layer scale."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(width, width, 7, padding=3, groups=width)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.pwconv1 = nn.Linear(width, 4 * width)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * width, width)
        self.gamma = nn.Parameter(torch.full((width,), 1e-6))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Return BCHW residual features, retaining input and parameter gradients."""
        y = self.norm(self.dwconv(x).permute(0, 2, 3, 1))
        y = self.pwconv2(self.act(self.pwconv1(y))) * self.gamma
        return x + y.permute(0, 3, 1, 2)

    forward = compute


class DinoConvNeXt(nn.Module):
    """Official weight keys, four native spatial stages and frozen evaluation mode.

    Input is ImageNet-normalized FP32 BCHW. `features` optionally applies the
    One-step mid-timestep diffusion L2 pool *inside* the hierarchy, rather than pooling returned features.
    Freezing parameters must not suppress gradients with respect to the image.
    """

    def __init__(self, widths: tuple[int, ...], depths: tuple[int, ...]) -> None:
        super().__init__()
        if len(widths) != 4 or len(depths) != 4 or min(*widths, *depths) <= 0:
            raise ValueError("DINO ConvNeXt requires four stages")
        self.embed_dims = widths
        self.recompute = False
        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(3, widths[0], 4, 4), ChannelLayerNorm(widths[0]))]
        )
        for left, right in pairwise(widths):
            self.downsample_layers.append(nn.Sequential(ChannelLayerNorm(left), nn.Conv2d(left, right, 2, 2)))
        self.stages = nn.ModuleList(
            nn.Sequential(*(ConvNeXtBlock(w) for _ in range(d))) for w, d in zip(widths, depths, strict=True)
        )
        self.norm: nn.LayerNorm | nn.Identity = nn.LayerNorm(widths[-1], eps=1e-6)
        self.norms = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity(), self.norm])

    def features(self, x: torch.Tensor, *, pooled: bool = False) -> tuple[torch.Tensor, ...]:
        """Return three BCHW levels; optional L2 pooling feeds each next stage.

        Input is normalized floating RGB BCHW, at least 16x16. Do not detach
        pixels: a frozen backbone remains differentiable to the generated image.
        """
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 16 or not x.is_floating_point():
            raise ValueError("DINO ConvNeXt requires floating RGB BCHW images at least 16x16")
        result = []
        for down, stage in zip(self.downsample_layers[:3], self.stages[:3], strict=True):
            # Frozen weights still retain large MLP activations for image gradients.
            # Checkpoint only this stateless backbone, never discriminator heads
            # whose spectral-normalization buffers change during forward.
            for module in (down, *stage):
                if self.recompute and torch.is_grad_enabled() and x.requires_grad:
                    x = checkpoint(module, x, use_reentrant=False, preserve_rng_state=False)
                else:
                    x = module(x)
            if pooled:
                x = l2_pool(x)
            result.append(x)
        return tuple(result)


def l2_pool(x: torch.Tensor) -> torch.Tensor:
    """Hanning(5)[1:-1] outer product, stride one, zero padding; BCHW -> BCHW."""
    window = x.new_tensor([0.5, 1.0, 0.5])
    kernel = (window[:, None] * window[None, :] / 4).expand(x.shape[1], 1, 3, 3)
    return (F.conv2d(x.square(), kernel, padding=1, groups=x.shape[1]) + 1e-12).sqrt()
