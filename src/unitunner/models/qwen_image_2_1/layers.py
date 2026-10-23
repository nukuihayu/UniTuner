# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Compute single-stream Qwen Image 2.1 modulation and native segmented attention."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unitunner.execution.attention import scaled_dot_product_attention
from unitunner.models.qwen_image_2_1.sequence import AttentionSegments


class ZeroCenteredRMSNorm(nn.Module):
    """Normalize last-axis variance in FP32 and apply learned (1+weight) scaling."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def normalize(self, values: Tensor) -> Tensor:
        """Return the input dtype with differentiable FP32 normalization intermediates."""
        values_float = values.float()
        return (
            values_float
            * torch.rsqrt(values_float.square().mean(-1, keepdim=True) + self.eps)
            * (1 + self.weight.float())
        ).to(values.dtype)

    forward = normalize


class TextProjection(nn.Module):
    """Normalize text then apply two bias-free projections with tanh-approximate GELU."""

    def __init__(self, context_width: int, width: int, eps: float) -> None:
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(context_width, eps)
        self.in_layer = nn.Linear(context_width, width, bias=False)
        self.out_layer = nn.Linear(width, width, bias=False)

    def project(self, values: Tensor) -> Tensor:
        """Map BLC text features to the shared BLD token stream."""
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(values)), approximate="tanh"))

    forward = project


class RMSNorm(nn.Module):
    """Preserve the published Q/K normalization's explicit cast and backward order."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def normalize(self, values: Tensor) -> Tensor:
        """Compute FP32 variance, then cast before the learned weight multiply."""
        normalized = values * torch.rsqrt(values.float().pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            normalized = normalized.to(self.weight.dtype)
        return normalized * self.weight

    forward = normalize


class TimestepEmbedding(nn.Module):
    """Own bias-free time projections with native serialized field names."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(256, width, bias=False)
        self.linear_2 = nn.Linear(width, width, bias=False)

    def project(self, values: Tensor) -> Tensor:
        """Apply SiLU between learned time projections."""
        return self.linear_2(F.silu(self.linear_1(values)))

    forward = project


class TimeEmbedding(nn.Module):
    """Project normalized flow time with scale1000 and native256 cos/sin features."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(width)

    def embed(self, timesteps: Tensor, dtype: torch.dtype) -> Tensor:
        """Return BD features using the published CPU FP32 frequency calculation."""
        # Rebuild this small constant rather than registering a buffer that .to(dtype)
        # would round, or meta-device loading would leave uninitialized.
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, device="cpu", dtype=torch.float32) / 128).to(
            timesteps.device
        )
        phase = 1000 * timesteps.float()[:, None] * frequencies[None]
        return self.timestep_embedder(torch.cat((phase.cos(), phase.sin()), dim=-1).to(dtype))

    forward = embed


class SwiGLU(nn.Module):
    """Compute bias-free out(SiLU(gate(x))*proj(x)) with native MLP weight keys."""

    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, hidden_width, bias=False)
        self.gate_layer = nn.Linear(width, hidden_width, bias=False)
        self.out = nn.Linear(hidden_width, width, bias=False)

    def project(self, values: Tensor) -> Tensor:
        """Return the original shape while retaining both gated gradient branches."""
        return self.out(F.silu(self.gate_layer(values)) * self.proj(values))

    forward = project


def apply_rotary(values: Tensor, rotary: Tensor) -> Tensor:
    """Rotate BLHD adjacent pairs in complex64, returning the input shape/dtype."""
    pairs = torch.view_as_complex(values.float().reshape(*values.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * rotary[:, :, None]).flatten(-2).to(values.dtype)


def segmented_attention(q: Tensor, k: Tensor, v: Tensor, segments: AttentionSegments) -> Tensor:
    """Attend BLHD segments with causal text and full current-image visibility.

    Image queries see all keys through their segment end; text queries see keys
    through their own position, including preceding images. Exclude right
    padding per sample, return padded BL(HD), preserve all Q/K/V gradients.
    Consecutive rows with identical visibility share batched kernels. Torch
    chooses its native SDPA backend; no global precision setting changes.
    """
    groups: list[tuple[int, int]] = []
    for index, runs in enumerate(segments):
        if index and runs == segments[index - 1]:
            first, _ = groups[-1]
            groups[-1] = (first, index + 1)
        else:
            groups.append((index, index + 1))
    rows = []
    for first, last in groups:
        runs = segments[first]
        parts = []
        for start, end, is_text in runs:
            mask = None
            if is_text and start:
                mask = torch.arange(end, device=q.device)[None] <= torch.arange(start, end, device=q.device)[:, None]
            result = (
                scaled_dot_product_attention(
                    q[first:last, start:end].transpose(1, 2),
                    k[first:last, :end].transpose(1, 2),
                    v[first:last, :end].transpose(1, 2),
                    attn_mask=mask,
                    is_causal=is_text and start == 0,
                )
                .transpose(1, 2)
                .flatten(2)
            )
            parts.append(result)
        row = torch.cat(parts, dim=1)
        padding = q.shape[1] - row.shape[1]
        rows.append(F.pad(row, (0, 0, 0, padding)) if padding else row)
    return rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)


class Attention(nn.Module):
    """Own bias-free Q/K/V/output and per-head learned RMSNorm projections."""

    def __init__(self, width: int, heads: int, head_dim: int, eps: float) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(width, width, bias=False)])
        self.norm_q = RMSNorm(head_dim, eps)
        self.norm_k = RMSNorm(head_dim, eps)

    def attend(self, values: Tensor, rotary: Tensor, segments: AttentionSegments) -> Tensor:
        """Return BLD image/text attention with matching Q/K/V autocast dtypes."""
        q = self.norm_q(self.to_q(values).unflatten(-1, (self.heads, -1)))
        k = self.norm_k(self.to_k(values).unflatten(-1, (self.heads, -1)))
        v = self.to_v(values).unflatten(-1, (self.heads, -1))
        return self.to_out[0](
            segmented_attention(apply_rotary(q.to(v.dtype), rotary), apply_rotary(k.to(v.dtype), rotary), v, segments)
        )

    forward = attend


class QwenImage21Block(nn.Module):
    """Apply target/zero-time scales and tanh-gated attention/MLP residuals."""

    def __init__(self, width: int, heads: int, head_dim: int, mlp_ratio: int, eps: float) -> None:
        super().__init__()
        self.img_norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=eps)
        self.attn = Attention(width, heads, head_dim, eps)
        self.img_norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLU(width, width * mlp_ratio)

    def transform(
        self, values: Tensor, modulation: Tensor, rotary: Tensor, segments: AttentionSegments, target_mask: Tensor
    ) -> Tensor:
        """Transform padded BLD, with B+1 modulation rows and a BL target mask."""
        scale1, gate1, scale2, gate2 = [
            torch.where(target_mask[..., None], part[:-1, None], part[-1:, None])
            for part in modulation.chunk(4, dim=-1)
        ]
        values = values + gate1.tanh() * self.attn(self.img_norm1(values) * (1 + scale1), rotary, segments)
        values = values + gate2.tanh() * self.img_mlp(self.img_norm2(values) * (1 + scale2))
        return values.clamp(-65504, 65504) if values.dtype == torch.float16 else values

    forward = transform


class FinalNorm(nn.Module):
    """Scale affine-free target LayerNorm using the sample's timestep features."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=eps)

    def normalize(self, values: Tensor, time_features: Tensor) -> Tensor:
        """Return BND target features with no conditioning tokens or additive shift."""
        # Select per-token scales before the multiply, as in the joint-token
        # reference. Broadcasting here instead changes BF16 backward rounding.
        scale = self.linear(F.silu(time_features).to(values.dtype))[:, None].expand_as(values)
        return self.norm(values) * (1 + scale)

    forward = normalize
