# Copyright (c) 2025 Z-Image Team. Apache-2.0.
# Adapted from https://github.com/Tongyi-MAI/Z-Image via Musubi Tuner.
# Modified to separate native computation from execution and training policies.
"""Compute Z-Image attention, modulation and FP32 normalization formulas."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unituner.execution.attention import scaled_dot_product_attention
from unituner.models.z_image.config.model import ZImageConfig


class TimestepEmbedder(nn.Module):
    """Project FP32 sinusoidal native timesteps through a learned MLP."""

    def __init__(self, config: ZImageConfig) -> None:
        super().__init__()
        self.frequency_dim = config.timestep_frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, config.timestep_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.timestep_hidden_dim, min(config.dim, 256)),
        )

    def embed(self, timesteps: Tensor) -> Tensor:
        """Embed native [B] timesteps (normally 0-1000), preserving MLP dtype."""
        half = self.frequency_dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device).float() / half)
        phases = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat((phases.cos(), phases.sin()), dim=-1)
        embedding = F.pad(embedding, (0, self.frequency_dim % 2))
        projection = self.mlp[0]
        if not isinstance(projection, nn.Linear):
            raise TypeError("Timestep projection must be Linear")
        return self.mlp(embedding.to(projection.weight.dtype))

    forward = embed


class RMSNorm(nn.Module):
    """Normalize the final axis in FP32, then restore the input dtype."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def normalize(self, x: Tensor) -> Tensor:
        """Return differentiable RMS-normalized values with learned FP32 scale."""
        values = x.float()
        return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(
            x.dtype
        )

    forward = normalize


def clamp_fp16(x: Tensor) -> Tensor:
    """Replace FP16 nonfinite values by the native saturation convention."""
    return torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504) if x.dtype == torch.float16 else x


def apply_rotary_embedding(x: Tensor, frequencies: Tensor) -> Tensor:
    """Rotate [B,L,H,D] pairs in FP32 using complex64 [B,L,D/2] phases."""
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * frequencies.unsqueeze(2)).flatten(3).to(x.dtype)


class FeedForward(nn.Module):
    """Compute native SwiGLU with the model's modulated FP16 downscale."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden_dim = int(dim / 3 * 8)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def project(self, x: Tensor, *, downscale: bool = False) -> Tensor:
        """Project [B,L,D]; divide the gate by 32 only for modulated FP16."""
        gate = self.w3(x)
        if downscale and x.dtype == torch.float16:
            gate = gate / 32
        return self.w2(clamp_fp16(F.silu(self.w1(x)) * gate))

    forward = project


class ZImageAttention(nn.Module):
    """Compute noncausal Torch SDPA; True mask entries denote active keys."""

    def __init__(self, config: ZImageConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.dim // config.num_heads
        self.to_q = nn.Linear(config.dim, config.dim, bias=False)
        self.to_k = nn.Linear(config.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(config.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(config.dim, config.dim, bias=False)])
        self.norm_q = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, config.norm_eps) if config.qk_norm else None

    def attend(self, x: Tensor, frequencies: Tensor, attention_mask: Tensor | None) -> Tensor:
        """Attend [B,L,D] tokens with [B,L] key validity and explicit RoPE."""
        q = self.to_q(x).unflatten(-1, (self.num_heads, self.head_dim))
        k = self.to_k(x).unflatten(-1, (self.num_kv_heads, self.head_dim))
        v = self.to_v(x).unflatten(-1, (self.num_kv_heads, self.head_dim))
        if self.norm_q is not None and self.norm_k is not None:
            q, k = self.norm_q(q), self.norm_k(k)
        q = apply_rotary_embedding(q, frequencies).to(v.dtype)
        k = apply_rotary_embedding(k, frequencies).to(v.dtype)
        mask = None if attention_mask is None else attention_mask[:, None, None, :]
        output = (
            scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=mask,
                dropout_p=0.0,
                enable_gqa=self.num_heads != self.num_kv_heads,
            )
            .transpose(1, 2)
            .flatten(2)
        )
        output = self.to_out[0](output)
        return output / 4 if output.dtype == torch.float16 else output

    forward = attend


class ZImageTransformerBlock(nn.Module):
    """Compute a native residual block, optionally conditioned by reverse time."""

    def __init__(self, config: ZImageConfig, *, modulation: bool) -> None:
        super().__init__()
        self.modulation = modulation
        self.attention = ZImageAttention(config)
        self.feed_forward = FeedForward(config.dim)
        self.attention_norm1 = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm2 = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm1 = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm2 = RMSNorm(config.dim, config.norm_eps)
        if modulation:
            self.adaLN_modulation = nn.ModuleList([nn.Linear(min(config.dim, 256), 4 * config.dim)])

    def transform(
        self, x: Tensor, frequencies: Tensor, conditioning: Tensor | None, attention_mask: Tensor | None
    ) -> Tensor:
        """Transform [B,L,D] without owning checkpoint, compile or transfer policy."""
        if self.modulation:
            if conditioning is None:
                raise ValueError("Modulated Z-Image blocks require timestep conditioning")
            scale_attention, gate_attention, scale_ffn, gate_ffn = (
                self.adaLN_modulation[0](conditioning).unsqueeze(1).chunk(4, -1)
            )
            attended = self.attention(self.attention_norm1(x) * (1 + scale_attention), frequencies, attention_mask)
            x = x + gate_attention.tanh() * self.attention_norm2(clamp_fp16(attended))
            projected = self.feed_forward(self.ffn_norm1(x) * (1 + scale_ffn), downscale=True)
            return x + gate_ffn.tanh() * self.ffn_norm2(clamp_fp16(projected))
        x = x + self.attention_norm2(clamp_fp16(self.attention(self.attention_norm1(x), frequencies, attention_mask)))
        return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

    forward = transform


class FinalLayer(nn.Module):
    """Project normalized and modulated tokens back to latent patches."""

    def __init__(self, dim: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(min(dim, 256), dim))

    def project(self, x: Tensor, conditioning: Tensor) -> Tensor:
        """Return [B,L,patch_volume*C] with differentiable time modulation."""
        return self.linear(self.norm_final(x) * (1 + self.adaLN_modulation(conditioning)).unsqueeze(1))

    forward = project


class RopeEmbedder:
    """Own CPU reference phase tables and a single explicit device cache.

    Tables are not model weights. Construct phases in FP64, round to FP32 before
    polar conversion, and preserve complex64 when model parameters change dtype.
    """

    def __init__(self, config: ZImageConfig) -> None:
        self.tables: tuple[Tensor, ...] = tuple(
            self._build(width, length, config.rope_theta)
            for width, length in zip(config.axes_dims, config.axes_lengths, strict=True)
        )
        self.device_tables: tuple[Tensor, ...] = self.tables
        self.device = torch.device("cpu")

    @staticmethod
    def _build(width: int, length: int, theta: float) -> Tensor:
        """Construct one complex64 axis table from FP64 frequencies."""
        frequencies = 1 / (theta ** (torch.arange(0, width, 2, dtype=torch.float64, device="cpu") / width))
        phases = torch.outer(torch.arange(length, dtype=torch.float64, device="cpu"), frequencies).float()
        return torch.polar(torch.ones_like(phases), phases)

    def __call__(self, position_ids: Tensor) -> Tensor:
        """Gather [B,L,3] integer coordinates into [B,L,head_dim/2] phases."""
        if position_ids.device != self.device:
            self.device_tables = tuple(table.to(position_ids.device) for table in self.tables)
            self.device = position_ids.device
        return torch.cat([table[position_ids[..., axis]] for axis, table in enumerate(self.device_tables)], dim=-1)
