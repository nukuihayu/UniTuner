# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Compute native FP32 Wan CLIP visual tokens without unused XLM-Roberta execution."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from unitunner.models.wan.config.clip import WanClipConfig


class ClipAttention(nn.Module):
    """Compute noncausal FP32 SDPA with native fused QKV order and biased projection."""

    def __init__(self, config: WanClipConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.to_qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def attend(self, x: Tensor) -> Tensor:
        """Map BLD FP32 tokens to BLD without precision conversion or a visibility mask."""
        batch, length, width = x.shape
        q, k, v = self.to_qkv(x).view(batch, length, 3, self.num_heads, width // self.num_heads).unbind(2)
        result = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(result.transpose(1, 2).contiguous().reshape(batch, length, width))

    forward = attend


class ClipBlock(nn.Module):
    """Apply native prenorm residual attention and exact GELU MLP with no dropout."""

    def __init__(self, config: WanClipConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = ClipAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * config.mlp_ratio),
            nn.GELU(),
            nn.Linear(config.hidden_size * config.mlp_ratio, config.hidden_size),
            nn.Dropout(0),
        )

    def refine(self, x: Tensor) -> Tensor:
        """Preserve native residual addition and layer-norm operation order."""
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

    forward = refine


class WanClipVision(nn.Module):
    """Own all visual weights; emit unpooled tokens before the last block.

    The final block, post_norm and projection head remain in the artifact but
    are intentionally inactive for Wan conditioning. FP32 CPU/CUDA only;
    callers choose SDPA backend and execution resources. No position resizing.
    """

    def __init__(self, config: WanClipConfig) -> None:
        super().__init__()
        self.config = config
        gain = config.hidden_size**-0.5
        self.patch_embedding = nn.Conv2d(3, config.hidden_size, config.patch_size, config.patch_size, bias=False)
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, config.hidden_size))
        self.pos_embedding = nn.Parameter(
            gain * torch.randn(1, (config.image_size // config.patch_size) ** 2 + 1, config.hidden_size)
        )
        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.transformer = nn.Sequential(*(ClipBlock(config) for _ in range(config.num_layers)))
        self.post_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.head = nn.Parameter(gain * torch.randn(config.hidden_size, config.output_dim))

    def encode(self, pixels: Tensor) -> Tensor:
        """Return BLD penultimate visual tokens from normalized B3SS FP32 pixels.

        Inputs are differentiable and not modified. Output includes the class
        token and all patches; no final norm, projection or pooling is applied.
        """
        if (
            pixels.ndim != 4
            or pixels.shape[0] < 1
            or pixels.shape[1:] != (3, self.config.image_size, self.config.image_size)
            or pixels.dtype != torch.float32
            or self.patch_embedding.weight.dtype != torch.float32
            or pixels.device != self.patch_embedding.weight.device
            or pixels.device.type not in ("cpu", "cuda")
        ):
            raise ValueError("Wan CLIP requires matching CPU/CUDA FP32 B3SS pixels and FP32 weights")
        if torch.is_autocast_enabled(pixels.device.type):
            raise ValueError("Wan CLIP core requires autocast disabled")
        x = self.patch_embedding(pixels).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.cls_embedding.expand(pixels.shape[0], -1, -1), x], dim=1)
        x = self.pre_norm(x + self.pos_embedding)
        for index in range(self.config.num_layers - 1):
            x = self.transformer[index](x)
        return x

    forward = encode
