# Copyright 2025-2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Diffusers PR #14355 (abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc)
# via the pinned Musubi and old MixTuner implementations.
"""MiniMax-H3 native time embeddings, attention, modulation and transformer layers."""

import math

import torch
from torch import Tensor, nn

from unituner.execution.attention import scaled_dot_product_attention
from unituner.models.minimax_h3.config.model import MiniMaxH3Config

_INT32_SAFE_EXTENT = 2**31


class TimeEmbedder(nn.Module):
    """Compute native FP32 cosine/sine model-time embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.proj_in = nn.Linear(input_dim, hidden_size, dtype=torch.float32, device=device)
        self.proj_out = nn.Linear(hidden_size, output_dim, dtype=torch.float32, device=device)
        self.activation = nn.SiLU()

    def embed(self, timesteps: Tensor) -> Tensor:
        """Embed one-dimensional native clean-fraction times in FP32."""
        half = self.input_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        angles = timesteps.to(torch.float32)[:, None] * frequencies[None]
        embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        with torch.autocast(device_type=timesteps.device.type, enabled=False):
            return self.proj_out(self.activation(self.proj_in(embedding)))

    forward = embed


class AdalnProj(nn.Module):
    """Project native time embeddings to ordered per-modality modulation rows."""

    def __init__(
        self,
        timestep_dim: int,
        hidden_size: int,
        expand: int,
        modalities: int,
        *,
        dtype: torch.dtype,
        apply_silu: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden_size = hidden_size
        # pruned-AdaLN checkpoints bake the activation into the curve table, so the
        # 8-wide interpolated embedding feeds the projection directly
        self.apply_silu = apply_silu
        self.linear = nn.Linear(
            timestep_dim,
            expand * hidden_size * modalities,
            dtype=dtype,
            device=device,
        )
        self.activation = nn.SiLU()

    def project(self, timestep_embeddings: Tensor) -> tuple[Tensor, ...]:
        if self.apply_silu:
            timestep_embeddings = self.activation(timestep_embeddings)
        projected = self.linear(timestep_embeddings)
        projected = projected.reshape(projected.shape[0] * self.modalities, self.expand * self.hidden_size)
        return projected.chunk(self.expand, dim=-1)

    forward = project


def _apply_rope_split_half(hidden_states: Tensor, rotation_table: Tensor) -> Tensor:
    """Rotate split-half channel pairs and preserve the unrotated tail."""
    pairs = rotation_table.shape[-3]
    rotary = torch.stack((hidden_states[..., :pairs], hidden_states[..., pairs : 2 * pairs]), dim=-1)
    rotary = torch.matmul(rotation_table, rotary.unsqueeze(-1)).squeeze(-1)
    rotary = torch.cat((rotary[..., 0], rotary[..., 1]), dim=-1)
    return torch.cat((rotary, hidden_states[..., 2 * pairs :]), dim=-1)


def _element_extent(tensor: Tensor) -> int | torch.SymInt:
    """Highest storage element index addressable through the view, plus one."""
    return (
        tensor.storage_offset()
        + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
        + 1
    )


# torch SDPA's memory-efficient backend computes block base addresses in int32, so a view
# whose element offsets pass 2^31 silently reads wrapped memory (verified: with the fused-qkv
# value view, every row from the first 128-aligned block past the line is corrupt, turning
# the whole sample into noise). The released 15 s maximum at 1344x768 packs ~109k rows and
# crosses the line at row 99,968.


class Attention(nn.Module):
    """Compute bias-free QKV attention with native RMS normalization and split-half RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        qk_norm_eps: float,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.qkv_proj = nn.Linear(hidden_size, self.inner_dim * 3, bias=False, dtype=dtype, device=device)
        self.q_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps, dtype=dtype, device=device)
        self.k_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps, dtype=dtype, device=device)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size, bias=False, dtype=dtype, device=device)

    def attend(self, hidden_states: Tensor, rotation_table: Tensor | None = None) -> Tensor:
        """Attend jointly over all supplied packed rows using Torch SDPA."""
        batch_size, sequence_length, _ = hidden_states.shape
        query, key, value = self.qkv_proj(hidden_states).split(self.inner_dim, dim=-1)
        query = self.q_norm(query.reshape(batch_size, sequence_length, self.num_heads, self.head_dim))
        key = self.k_norm(key.reshape(batch_size, sequence_length, self.num_heads, self.head_dim))
        value = value.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        # query/key leave the RMSNorms as fresh contiguous tensors, but value is still a view
        # into the fused qkv buffer (sequence stride 3 * inner_dim): materialize it before its
        # offsets outgrow int32 address math in the SDPA backends
        if _element_extent(value) > _INT32_SAFE_EXTENT:
            value = value.contiguous()
        if rotation_table is not None:
            query = _apply_rope_split_half(query, rotation_table)
            key = _apply_rope_split_half(key, rotation_table)
        query = query.to(value)
        key = key.to(value)

        output = (
            scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            )
            .transpose(1, 2)
            .reshape(batch_size, sequence_length, self.inner_dim)
        )
        return self.out_proj(output)

    forward = attend


class MLP(nn.Module):
    """Compute the native SiLU-gated feed-forward projection."""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size * 2, bias=False, dtype=dtype, device=device)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.activation = nn.SiLU()

    def project(self, hidden_states: Tensor) -> Tensor:
        gate, values = self.fc1(hidden_states).chunk(2, dim=-1)
        return self.fc2(self.activation(gate) * values)

    forward = project


class RefinerBlock(nn.Module):
    """Refine text tokens with unmodulated attention and gated MLP residuals."""

    def __init__(
        self,
        config: MiniMaxH3Config,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device)
        self.attn = Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.mlp = MLP(config.hidden_size, config.ffn_hidden_size, dtype=dtype, device=device)

    def refine(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))

    forward = refine


class TokenRefiner(nn.Module):
    """Apply text refiner blocks and final native RMS normalization."""

    def __init__(
        self,
        config: MiniMaxH3Config,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [RefinerBlock(config, dtype=dtype, device=device) for _ in range(config.token_refiner_num_layers)]
        )
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps, dtype=dtype, device=device)

    def refine(self, hidden_states: Tensor) -> Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)

    forward = refine


def _mod_scale_shift(
    hidden_states: Tensor,
    shift: Tensor,
    scale: Tensor,
    segments: tuple[tuple[int, int, int], ...],
) -> Tensor:
    """Modulate fresh [B,L,D] norm output using [R,D] shift/scale rows.

    Segments are disjoint (start, stop, row) token ranges. Updates alias the
    fresh normalization output, never the residual. Native CopySlices backward
    preserves BF16 accumulation into the shared trainable timestep embedding.
    """
    for start, stop, row in segments:
        hidden_states[:, start:stop].mul_(1.0 + scale[row]).add_(shift[row])
    return hidden_states


def _mod_gate(
    residual: Tensor,
    update: Tensor,
    gate: Tensor,
    segments: tuple[tuple[int, int, int], ...],
) -> Tensor:
    """Add gated [B,L,D] updates to a fresh copy of the residual.

    Gate rows [R,D] follow disjoint (start, stop, row) segments. Both inputs
    retain gradients; native slice-update backward preserves eager rounding.
    """
    output = residual.clone()
    for start, stop, row in segments:
        output[:, start:stop].add_(update[:, start:stop] * gate[row])
    return output


class DiTBlock(nn.Module):
    """Apply three-modality AdaLN attention and MLP residual updates."""

    def __init__(
        self,
        config: MiniMaxH3Config,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device)
        self.attn = Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.mlp = MLP(config.hidden_size, config.ffn_hidden_size, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(
            config.adaln_in_features,
            config.hidden_size,
            expand=6,
            modalities=3,
            dtype=dtype,
            apply_silu=not config.is_pruned,
            device=device,
        )

    def refine(
        self,
        hidden_states: Tensor,
        timestep_embeddings: Tensor,
        modulation_segments: tuple[tuple[int, int, int], ...],
        rotation_table: Tensor,
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(timestep_embeddings)
        normalized = _mod_scale_shift(self.norm1(hidden_states), shift_msa, scale_msa, modulation_segments)
        hidden_states = _mod_gate(
            hidden_states,
            self.attn(normalized, rotation_table),
            gate_msa,
            modulation_segments,
        )
        normalized = _mod_scale_shift(self.norm2(hidden_states), shift_mlp, scale_mlp, modulation_segments)
        return _mod_gate(hidden_states, self.mlp(normalized), gate_mlp, modulation_segments)

    forward = refine


class FinalLayer(nn.Module):
    """Project target video/audio streams through their native time rows."""

    def __init__(
        self,
        hidden_size: int,
        timestep_dim: int,
        video_output_dim: int,
        audio_output_dim: int,
        *,
        norm_eps: float = 1e-5,
        dtype: torch.dtype,
        apply_silu: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=norm_eps, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(
            timestep_dim,
            hidden_size,
            expand=2,
            modalities=1,
            dtype=dtype,
            apply_silu=apply_silu,
            device=device,
        )
        self.video_out = nn.Linear(hidden_size, video_output_dim, dtype=torch.float32, device=device)
        self.audio_out = nn.Linear(hidden_size, audio_output_dim, dtype=torch.float32, device=device)

    def project(
        self,
        hidden_states: Tensor,
        timestep_embeddings: Tensor,
        *,
        video_slice: slice,
        audio_slice: slice,
        video_timestep_index: int,
        audio_timestep_index: int,
    ) -> tuple[Tensor, Tensor]:
        shift, scale = self.adaln_proj(timestep_embeddings)
        video = self.norm(hidden_states[:, video_slice])
        video = video * (1.0 + scale[video_timestep_index]) + shift[video_timestep_index]
        audio = self.norm(hidden_states[:, audio_slice])
        audio = audio * (1.0 + scale[audio_timestep_index]) + shift[audio_timestep_index]
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return self.video_out(video.to(torch.float32)), self.audio_out(audio.to(torch.float32))

    forward = project


class RotationFrequencies(nn.Module):
    """Checkpoint-owned FP32 rotary frequencies without changing the rope.inv_freq key."""

    inv_freq: Tensor

    def __init__(self, length: int, device: torch.device | str | None) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq", 10000.0 ** (-torch.arange(length, dtype=torch.float32, device=device) / length)
        )
