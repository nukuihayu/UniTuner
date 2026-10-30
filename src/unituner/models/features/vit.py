# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from facebookresearch/dinov3; see licenses/DINOv3-LICENSE.md.
"""Compute frozen DINOv3 MLP ViT patch features without runtime upstream imports."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class DinoViTConfig:
    """Describe published MLP ViT geometry or an explicit compact fixture."""

    width: int = 768
    depth: int = 12
    heads: int = 12
    patch: int = 16
    registers: int = 4
    image_size: int = 224
    hidden: int = 3072

    def __post_init__(self) -> None:
        if min(self.width, self.depth, self.heads, self.patch, self.image_size, self.hidden) <= 0:
            raise ValueError("DINO ViT dimensions must be positive")
        if self.width % (4 * self.heads) or self.registers < 0 or self.image_size % self.patch:
            raise ValueError("DINO ViT requires axial head dimensions and a divisible image grid")


class MaskedQKV(nn.Linear):
    """Mask key bias while preserving published fused QKV keys."""

    bias_mask: torch.Tensor

    def __init__(self, width: int) -> None:
        super().__init__(width, 3 * width)
        mask = torch.ones(3 * width)
        mask[width : 2 * width] = 0
        self.register_buffer("bias_mask", mask)

    def compute(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias * self.bias_mask)

    forward = compute


class DinoAttention(nn.Module):
    """Apply native two-axis RoPE to patches, excluding prefix tokens."""

    def __init__(self, config: DinoViTConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.qkv = MaskedQKV(config.width)
        self.proj = nn.Linear(config.width, config.width)

    def compute(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        prefix = n - sin.shape[0]
        rotated = []
        for value in (q, k):
            patches = value[:, :, prefix:].float()
            left, right = patches.chunk(2, -1)
            patches = patches * cos + torch.cat((-right, left), -1) * sin
            rotated.append(torch.cat((value[:, :, :prefix], patches.to(value.dtype)), 2))
        result = F.scaled_dot_product_attention(rotated[0], rotated[1], v)
        return self.proj(result.transpose(1, 2).reshape(b, n, c))

    forward = compute


class LayerScale(nn.Module):
    """Scale each residual channel with a learned constant."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((width,), 1e-5))

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma

    forward = compute


class DinoMLP(nn.Module):
    """Apply the native GELU feed-forward branch."""

    def __init__(self, config: DinoViTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.width, config.hidden)
        self.fc2 = nn.Linear(config.hidden, config.width)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

    forward = compute


class DinoBlock(nn.Module):
    """Compute frozen attention and MLP residuals."""

    def __init__(self, config: DinoViTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width, eps=1e-5)
        self.attn = DinoAttention(config)
        self.ls1 = LayerScale(config.width)
        self.norm2 = nn.LayerNorm(config.width, eps=1e-5)
        self.mlp = DinoMLP(config)
        self.ls2 = LayerScale(config.width)

    def compute(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), sin, cos))
        return x + self.ls2(self.mlp(self.norm2(x)))

    forward = compute


class PatchEmbed(nn.Module):
    """Map nonoverlapping RGB patches to row-major tokens."""

    def __init__(self, config: DinoViTConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, config.width, config.patch, config.patch)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)

    forward = compute


class DinoRope(nn.Module):
    """Build normalized spatial rotary phases for patch tokens."""

    periods: torch.Tensor

    def __init__(self, config: DinoViTConfig) -> None:
        super().__init__()
        dim = config.width // config.heads
        self.register_buffer("periods", 100 ** (2 * torch.arange(dim // 4).float() / (dim // 2)))

    def compute(self, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.periods.device
        y = torch.arange(0.5, h, device=device) / h
        x = torch.arange(0.5, w, device=device) / w
        coords = torch.stack(torch.meshgrid(y, x, indexing="ij"), -1).flatten(0, 1) * 2 - 1
        angles = (2 * math.pi * coords[:, :, None] / self.periods.float()).flatten(1).tile(2)
        return angles.sin(), angles.cos()

    forward = compute


class DinoViT(nn.Module):
    """DINOv3 evaluation ViT with official RoPE, masked K bias and register keys."""

    def __init__(self, config: DinoViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.width))
        if config.registers:
            self.storage_tokens = nn.Parameter(torch.zeros(1, config.registers, config.width))
        self.mask_token = nn.Parameter(torch.zeros(1, config.width))
        self.rope_embed = DinoRope(config)
        self.blocks = nn.ModuleList(DinoBlock(config) for _ in range(config.depth))
        self.norm = nn.LayerNorm(config.width, eps=1e-5)

    def compute(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return normalized patch tokens [B,HW/patch²,E], excluding CLS/registers."""
        h, w = pixels.shape[-2:]
        if h % self.config.patch or w % self.config.patch:
            raise ValueError("DINO input dimensions must be divisible by its patch size")
        patches = self.patch_embed(pixels)
        prefix = [self.cls_token.expand(pixels.shape[0], -1, -1)]
        if self.config.registers:
            prefix.append(self.storage_tokens.expand(pixels.shape[0], -1, -1))
        x = torch.cat((*prefix, patches), 1)
        sin, cos = self.rope_embed(h // self.config.patch, w // self.config.patch)
        for block in self.blocks:
            x = block(x, sin, cos)
        return self.norm(x)[:, 1 + self.config.registers :]

    forward = compute
