# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Compute Wan modulation and native BF16-attention paths with unchanged weight keys."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unitunner.execution.attention import scaled_dot_product_attention
from unitunner.models.wan.config.model import WanConfig
from unitunner.models.wan.geometry import rotate


def attention(q: Tensor, k: Tensor, v: Tensor, lengths: tuple[int, ...] | None = None) -> Tensor:
    """Return BLHD SDPA, preserving native conversion of non-half Q/K/V to BF16.

    Only keys are cropped for ragged self attention; padded queries still attend
    to valid keys, matching the old public Torch path. Output uses query dtype.
    No RCM mask is constructed or consumed.
    """
    output_dtype = q.dtype
    q, k, v = (value if value.dtype in (torch.float16, torch.bfloat16) else value.bfloat16() for value in (q, k, v))
    q, k, v = (value.transpose(1, 2) for value in (q, k, v))
    if lengths is None or all(length == k.shape[2] for length in lengths):
        result = scaled_dot_product_attention(q, k, v)
    else:
        result = torch.cat(
            [
                scaled_dot_product_attention(
                    q[index : index + 1], k[index : index + 1, :, :length], v[index : index + 1, :, :length]
                )
                for index, length in enumerate(lengths)
            ]
        )
    return result.transpose(1, 2).contiguous().to(output_dtype)


class RMSNorm(nn.Module):
    """Normalize across the full projection width before splitting heads."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def normalize(self, values: Tensor) -> Tensor:
        """Reduce squares in FP32, restore input dtype, then multiply native scale."""
        # The reference casts once before entering _norm. Sharing this branch
        # accumulates in FP32 before the input gradient is rounded to BF16.
        floating = values.float()
        normalized = floating * torch.rsqrt(floating.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(values.dtype) * self.weight.to(values.dtype)

    forward = normalize


class LayerNorm(nn.LayerNorm):
    """Compute native layer normalization in FP32, returning the input dtype."""

    def normalize(self, input: Tensor) -> Tensor:
        """Use native affine parameters and FP32 reduction without changing keys."""
        return F.layer_norm(
            input.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        ).to(input.dtype)

    forward = normalize


class Attention(nn.Module):
    """Own separate native Q/K/V/O projections and optional image cross branch."""

    def __init__(self, config: WanConfig, *, image_context: bool = False) -> None:
        super().__init__()
        width = config.hidden_size
        self.num_heads, self.head_dim = config.num_heads, width // config.num_heads
        self.image_context = image_context
        self.image_token_count = config.image_token_count if image_context else 0
        self.q, self.k, self.v, self.o = (nn.Linear(width, width) for _ in range(4))
        self.norm_q = RMSNorm(width, config.norm_eps) if config.qk_norm else nn.Identity()
        self.norm_k = RMSNorm(width, config.norm_eps) if config.qk_norm else nn.Identity()
        if image_context:
            self.k_img, self.v_img = nn.Linear(width, width), nn.Linear(width, width)
            self.norm_k_img = RMSNorm(width, config.norm_eps) if config.qk_norm else nn.Identity()

    def attend(self, values: Tensor, context: Tensor | None, rotations: tuple[Tensor, ...] | None) -> Tensor:
        """Apply rotary self attention or independent text/image cross attention."""
        # Captions are padded to the model's fixed text_length and the image
        # prefix is fixed too. Specialize only this constant axis: Inductor 2.13
        # otherwise fails to split fused loops over (context_length - 257/514).
        # Video token count and batch size remain dynamic.
        if context is not None and torch.compiler.is_compiling():
            torch._dynamo.mark_static(context, 1)
        batch = values.shape[0]
        q = self.norm_q(self.q(values)).view(batch, -1, self.num_heads, self.head_dim)
        source = values if context is None else context
        image = None
        if self.image_context:
            image, source = source[:, : self.image_token_count], source[:, self.image_token_count :]
        k = self.norm_k(self.k(source)).view(batch, -1, self.num_heads, self.head_dim)
        v = self.v(source).view(batch, -1, self.num_heads, self.head_dim)
        lengths = None
        if rotations is not None:
            q, k = rotate(q, rotations), rotate(k, rotations)
            lengths = tuple(rotation.shape[0] for rotation in rotations)
        result = attention(q, k, v, lengths).flatten(2)
        if image is not None:
            k = self.norm_k_img(self.k_img(image)).view(batch, -1, self.num_heads, self.head_dim)
            v = self.v_img(image).view(batch, -1, self.num_heads, self.head_dim)
            result = result + attention(q, k, v).flatten(2)
        return self.o(result)

    forward = attend


class WanBlock(nn.Module):
    """Apply six-way time modulation, self attention, cross attention and tanh GELU."""

    def __init__(self, config: WanConfig) -> None:
        super().__init__()
        width = config.hidden_size
        self.version = config.version
        self.norm1 = LayerNorm(width, config.norm_eps, elementwise_affine=False)
        self.self_attn = Attention(config)
        self.norm3 = LayerNorm(width, config.norm_eps) if config.cross_attention_norm else nn.Identity()
        self.cross_attn = Attention(config, image_context=config.image_context)
        self.norm2 = LayerNorm(width, config.norm_eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(width, config.ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(config.ffn_dim, width)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, width) / width**0.5)

    def refine(self, values: Tensor, time: Tensor, context: Tensor, rotations: tuple[Tensor, ...]) -> Tensor:
        """Preserve native FP32 addcmul ordering for either batch or token time."""
        dtype = values.dtype
        if self.version == "2.1":
            shift, scale, gate, postshift, postscale, postgate = (self.modulation.float() + time).chunk(6, dim=1)
        else:
            shift, scale, gate, postshift, postscale, postgate = (
                part.squeeze(2) for part in (self.modulation.float().unsqueeze(0) + time).chunk(6, dim=2)
            )
        y = self.self_attn(torch.addcmul(shift, self.norm1(values).float(), 1 + scale).to(dtype), None, rotations)
        values = torch.addcmul(values, y.float(), gate).to(dtype)
        values = values + self.cross_attn(self.norm3(values), context, None)
        y = self.ffn(torch.addcmul(postshift, self.norm2(values).float(), 1 + postscale).to(dtype))
        return torch.addcmul(values, y.float(), postgate).to(dtype)

    forward = refine


class Head(nn.Module):
    """Normalize, modulate and project native F/H/W patch predictions."""

    def __init__(self, config: WanConfig) -> None:
        super().__init__()
        self.version = config.version
        self.norm = LayerNorm(config.hidden_size, config.norm_eps, elementwise_affine=False)
        self.head = nn.Linear(config.hidden_size, math.prod(config.patch_size) * config.out_channels)
        self.modulation = nn.Parameter(torch.randn(1, 2, config.hidden_size) / config.hidden_size**0.5)

    def project(self, values: Tensor, time: Tensor) -> Tensor:
        """Retain native FP32 final modulation with shift first, scale second."""
        if self.version == "2.1":
            shift, scale = (self.modulation.float() + time.unsqueeze(1)).chunk(2, dim=1)
        else:
            shift, scale = (
                part.squeeze(2) for part in (self.modulation.float().unsqueeze(0) + time.unsqueeze(2)).chunk(2, dim=2)
            )
        return self.head(torch.addcmul(shift, self.norm(values), 1 + scale))

    forward = project


class ImageProjection(nn.Module):
    """Project native CLIP tokens with an optional learned first/last position embedding."""

    def __init__(self, width: int, *, first_last_frame: bool = False) -> None:
        super().__init__()
        self.first_last_frame = first_last_frame
        self.proj = nn.Sequential(
            nn.LayerNorm(1280), nn.Linear(1280, 1280), nn.GELU(), nn.Linear(1280, width), nn.LayerNorm(width)
        )
        if first_last_frame:
            self.emb_pos = nn.Parameter(torch.zeros(1, 514, 1280))

    def project(self, values: Tensor) -> Tensor:
        """Project FP32 B2571280 or B22571280 features to B257D or B514D.

        First/last images are consecutive within each batch row. Add the native
        learned position embedding before LayerNorm/projection, preserving the
        official FLF2V order. Gradients reach both images and all parameters.
        The owning transformer validates the shape/device/dtype boundary.
        """
        if self.first_last_frame:
            values = values.reshape(-1, 514, 1280) + self.emb_pos
        return self.proj(values)

    forward = project
