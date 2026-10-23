# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See licenses/Apache-2.0.txt.
"""Compute native Qwen Image dual-stream attention and time modulation."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unitunner.execution.attention import scaled_dot_product_attention
from unitunner.models.qwen_image.positions import apply_rotary


class RMSNorm(nn.Module):
    """Apply learned RMS normalization using native FP32 variance and epsilon1e-6."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def normalize(self, values: Tensor) -> Tensor:
        """Normalize the last axis with FP32 variance and native weight dtype policy."""
        normalized = values * torch.rsqrt(values.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            normalized = normalized.to(self.weight.dtype)
        return normalized * self.weight

    forward = normalize


def timestep_embedding(timesteps: Tensor) -> Tensor:
    """Embed normalized B flow times with native256 cos/sin channels and scale1000."""
    exponent = -math.log(10000) * torch.arange(128, dtype=torch.float32, device=timesteps.device)
    frequencies = torch.exp(exponent / 128)
    angles = 1000 * (timesteps[:, None].float() * frequencies[None])
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


class TimestepEmbedding(nn.Module):
    """Project the fixed256 time features, preserving published parameter keys."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(256, width)
        self.linear_2 = nn.Linear(width, width)

    def project(self, values: Tensor) -> Tensor:
        """Apply SiLU between the two learned projections."""
        return self.linear_2(F.silu(self.linear_1(values)))

    forward = project


class QwenTimeEmbedding(nn.Module):
    """Own native time projections and optional layered RGB/RGBA embedding."""

    def __init__(self, width: int, *, layered: bool) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(width)
        self.addition_t_embedding: nn.Embedding | None = nn.Embedding(2, width) if layered else None

    def embed(self, timesteps: Tensor, dtype: torch.dtype, condition: Tensor | None) -> Tensor:
        """Return BD time vectors; condition validation belongs to the model boundary."""
        result = self.timestep_embedder(timestep_embedding(timesteps).to(dtype))
        if self.addition_t_embedding is not None:
            if condition is None:
                raise ValueError("Layered time embedding requires RGB/RGBA condition")
            result = result + self.addition_t_embedding(condition).to(dtype)
        return result

    forward = embed


class AdaptiveOutputNorm(nn.Module):
    """Apply the checkpoint's final SiLU scale/shift projection and affine-free norm."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, 2 * width)
        self.norm = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)

    def normalize(self, values: Tensor, conditioning: Tensor) -> Tensor:
        """Normalize BLD and modulate with BD time vectors."""
        scale, shift = self.linear(F.silu(conditioning).to(values.dtype)).chunk(2, dim=1)
        return self.norm(values) * (1 + scale)[:, None] + shift[:, None]

    forward = normalize


class GELUProjection(nn.Module):
    """Preserve FFN net.0.proj keys and native tanh-approximate GELU."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, 4 * width)

    def project(self, values: Tensor) -> Tensor:
        """Project and apply the checkpoint's approximate GELU."""
        return F.gelu(self.proj(values), approximate="tanh")

    forward = project


class FeedForward(nn.Module):
    """Compute native four-times-width FFN with published net keys."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(GELUProjection(width), nn.Dropout(0), nn.Linear(4 * width, width))

    def project(self, values: Tensor) -> Tensor:
        """Return a same-shape differentiable stream projection."""
        return self.net(values)

    forward = project


class JointAttention(nn.Module):
    """Attend image and text streams with separate native biased QKV projections."""

    def __init__(self, width: int, heads: int, *, output_text: bool) -> None:
        super().__init__()
        self.heads = heads
        self.output_text = output_text
        self.norm_q = RMSNorm(width // heads)
        self.norm_k = RMSNorm(width // heads)
        self.norm_added_q = RMSNorm(width // heads)
        self.norm_added_k = RMSNorm(width // heads)
        self.to_q = nn.Linear(width, width)
        self.to_k = nn.Linear(width, width)
        self.to_v = nn.Linear(width, width)
        self.add_q_proj = nn.Linear(width, width)
        self.add_k_proj = nn.Linear(width, width)
        self.add_v_proj = nn.Linear(width, width)
        self.to_out = nn.Sequential(nn.Linear(width, width), nn.Identity())
        self.to_add_out = nn.Linear(width, width)

    def attend(
        self, image: Tensor, text: Tensor, mask: Tensor | None, image_rope: Tensor, text_rope: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return image/text BLD attention with boolean text-key padding exclusion.

        All image keys are valid. Text padding remains a query but cannot become
        an attended key. SDPA is noncausal with no dropout. Final-block unused
        text output projections execute without autograd, preserving None rather
        than artificial zero gradients and avoiding unintended weight decay.
        """
        iq = self.norm_q(self.to_q(image).unflatten(-1, (self.heads, -1)))
        ik = self.norm_k(self.to_k(image).unflatten(-1, (self.heads, -1)))
        iv = self.to_v(image).unflatten(-1, (self.heads, -1))
        tq = self.norm_added_q(self.add_q_proj(text).unflatten(-1, (self.heads, -1)))
        tk = self.norm_added_k(self.add_k_proj(text).unflatten(-1, (self.heads, -1)))
        tv = self.add_v_proj(text).unflatten(-1, (self.heads, -1))
        query = torch.cat((apply_rotary(iq, image_rope), apply_rotary(tq, text_rope)), dim=1)
        key = torch.cat((apply_rotary(ik, image_rope), apply_rotary(tk, text_rope)), dim=1)
        value = torch.cat((iv, tv), dim=1)
        attention_mask = None
        if mask is not None:
            attention_mask = torch.cat((mask.new_ones((mask.shape[0], image.shape[1])), mask), dim=1)[:, None, None]
        attended = (
            scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attention_mask
            )
            .transpose(1, 2)
            .flatten(2)
            .to(query.dtype)
        )
        image_output = self.to_out(attended[:, : image.shape[1]])
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.output_text):
            text_output = self.to_add_out(attended[:, image.shape[1] :])
        return image_output, text_output

    forward = attend


def modulate(values: Tensor, parameters: Tensor, zero_index: int | None = None) -> tuple[Tensor, Tensor]:
    """Apply shift/scale/gate, optionally splitting target versus zero-time references."""
    shift, scale, gate = parameters.chunk(3, dim=-1)
    if zero_index is None:
        return values * (1 + scale[:, None]) + shift[:, None], gate[:, None]
    shift_base, shift_reference = shift.chunk(2)
    scale_base, scale_reference = scale.chunk(2)
    gate_base, gate_reference = gate.chunk(2)
    result = torch.cat(
        (
            values[:, :zero_index] * (1 + scale_base[:, None]) + shift_base[:, None],
            values[:, zero_index:] * (1 + scale_reference[:, None]) + shift_reference[:, None],
        ),
        dim=1,
    )
    gates = torch.cat(
        (
            gate_base[:, None].expand(-1, zero_index, -1),
            gate_reference[:, None].expand(-1, values.shape[1] - zero_index, -1),
        ),
        dim=1,
    )
    return result, gates


class QwenImageBlock(nn.Module):
    """Compute joint attention and gated residuals without execution-policy ownership."""

    def __init__(self, width: int, heads: int, *, zero_time: bool, output_text: bool) -> None:
        super().__init__()
        self.zero_time = zero_time
        self.output_text = output_text
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.img_norm1 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.img_norm2 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.txt_norm1 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.txt_norm2 = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.attn = JointAttention(width, heads, output_text=output_text)
        self.img_mlp = FeedForward(width)
        self.txt_mlp = FeedForward(width)

    def transform(
        self,
        image: Tensor,
        text: Tensor,
        mask: Tensor | None,
        time: Tensor,
        image_rope: Tensor,
        text_rope: Tensor,
        zero_index: int | None,
    ) -> tuple[Tensor, Tensor]:
        """Return (text,image) residual streams, preserving the final unused text graph."""
        im1, im2 = self.img_mod(time).chunk(2, dim=-1)
        tm1, tm2 = self.txt_mod(time.chunk(2)[0] if self.zero_time else time).chunk(2, dim=-1)
        image_values, image_gate = modulate(self.img_norm1(image), im1, zero_index)
        text_values, text_gate = modulate(self.txt_norm1(text), tm1)
        image_attended, text_attended = self.attn(image_values, text_values, mask, image_rope, text_rope)
        image = torch.addcmul(image, image_gate, image_attended)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.output_text):
            text = torch.addcmul(text, text_gate, text_attended)
        image_values, image_gate = modulate(self.img_norm2(image), im2, zero_index)
        image = torch.addcmul(image, image_gate, self.img_mlp(image_values))
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.output_text):
            text_values, text_gate = modulate(self.txt_norm2(text), tm2)
            text = torch.addcmul(text, text_gate, self.txt_mlp(text_values))
        if image.dtype == torch.float16:
            image = image.clamp(-65504, 65504)
        if text.dtype == torch.float16:
            text = text.clamp(-65504, 65504)
        return text, image

    forward = transform
