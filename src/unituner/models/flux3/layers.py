# Derived from Black Forest Labs FLUX3 Image Klein, Apache-2.0.
"""Compute native bias-free FLUX3 attention/SwiGLU blocks without execution policies."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from unituner.execution.attention import scaled_dot_product_attention
from unituner.models.flux3.conditioning import Flux3Bounds

type Flux3Modulation = tuple[Tensor, Tensor, Tensor]


def timestep_embedding(timesteps: Tensor) -> Tensor:
    """Embed L noise fractions into L256 features using native FP32 time*1000 phases."""
    scaled = 1000.0 * timesteps
    frequencies = torch.exp(-math.log(10000) * torch.arange(128, device=timesteps.device, dtype=torch.float32) / 128)
    phases = scaled[:, None].float() * frequencies[None]
    return torch.cat((phases.cos(), phases.sin()), dim=-1).to(timesteps)


def rope_embed(ids: Tensor, axes: tuple[int, int, int, int], theta: int) -> Tensor:
    """Return B1L(D/2)22 native matrices from BL4 positions, using FP64 phases."""
    with torch.autocast(ids.device.type, enabled=False):
        matrices = []
        for index, width in enumerate(axes):
            scale = torch.arange(0, width, 2, dtype=torch.float64, device=ids.device) / width
            phases = torch.einsum("...n,d->...nd", ids[..., index], 1.0 / (theta**scale))
            rotations = torch.stack((phases.cos(), -phases.sin(), phases.sin(), phases.cos()), dim=-1)
            matrices.append(rotations.reshape(*phases.shape, 2, 2).float())
        return torch.cat(matrices, dim=-3).unsqueeze(1)


def rmsnorm_rope(values: Tensor, scale: Tensor, frequencies: Tensor) -> Tensor:
    """Normalize BHLD heads and rotate interleaved pairs in FP32, then restore dtype."""
    with torch.autocast(values.device.type, enabled=False):
        batch, heads, length, width = values.shape
        floating = values.float()
        normalized = floating * floating.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt() * scale.float()
        pairs = normalized.reshape(batch, heads, length, width // 2, 1, 2)
        rotations = frequencies.float()
        rotated = rotations[..., 0] * pairs[..., 0] + rotations[..., 1] * pairs[..., 1]
        return rotated.reshape_as(values).to(values.dtype)


def layernorm_modulation(values: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """Apply native nonaffine LayerNorm (epsilon 1e-6), then scale and shift LD rows."""
    return (1.0 + scale) * F.layer_norm(values, (values.shape[-1],), eps=1e-6) + shift


def packed_attention(q: Tensor, k: Tensor, v: Tensor, bounds: Flux3Bounds) -> Tensor:
    """Return LD noncausal attention, isolating each contiguous sample without batch padding.

    Inputs are 1HLD; empty bounds perform no attention. Backend selection belongs
    to the caller's Torch context. No private global policy, dropout or implicit
    reference masking occurs. Unused phantom tokens have no attention dependency.
    """
    _, heads, length, width = q.shape
    result = q.new_zeros(length, heads * width)
    for start, end in bounds:
        if end > start:
            attended = scaled_dot_product_attention(q[:, :, start:end], k[:, :, start:end], v[:, :, start:end])
            result[start:end] = attended[0].transpose(0, 1).reshape(end - start, heads * width)
    return result


class RMSNorm(nn.Module):
    """Own the official head scale; fused normalization/rotation applies its value."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))


class QKNorm(nn.Module):
    """Own independent query/key head scales at published tensor keys."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.query_norm = RMSNorm(width)
        self.key_norm = RMSNorm(width)

    def normalize(self, q: Tensor, k: Tensor, frequencies: Tensor) -> tuple[Tensor, Tensor]:
        """Return native normalized/rotated BHLD queries and keys, retaining all gradients."""
        return rmsnorm_rope(q, self.query_norm.scale, frequencies), rmsnorm_rope(k, self.key_norm.scale, frequencies)

    forward = normalize


class MLPEmbedder(nn.Module):
    """Own native bias-free SiLU time projections at published parameter keys."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def project(self, values: Tensor) -> Tensor:
        """Return differentiable LD features without mode or RNG mutation."""
        return self.out_layer(F.silu(self.in_layer(values)))

    forward = project


class ModeModulation(nn.Module):
    """Produce one LD shift, scale and residual gate for each native token row."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin = nn.Linear(width, 3 * width, bias=False)

    def modulate(self, vector: Tensor) -> Flux3Modulation:
        """Return shared-per-stream modulation without copying or detaching rows."""
        shift, scale, gate = self.lin(F.silu(vector)).chunk(3, dim=-1)
        return shift, scale, gate

    forward = modulate


class LastLayer(nn.Module):
    """Project target tokens after native adaptive nonaffine final normalization."""

    def __init__(self, hidden: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False))

    def project(self, values: Tensor, vector: Tensor) -> Tensor:
        """Return LD native velocity while retaining token/time/parameter gradients."""
        shift, scale = self.adaLN_modulation(vector).chunk(2, dim=-1)
        return self.linear(layernorm_modulation(values, scale, shift))

    forward = project


class Flux3Block(nn.Module):
    """Own parallel attention and SwiGLU projections with official bias-free keys.

    The same block serves early independent modes and shared sample-major tokens.
    Recompute, compilation, feature capture and residency are caller-owned.
    """

    def __init__(self, hidden: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.heads = heads
        mlp_hidden = int(hidden * mlp_ratio)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, 2 * mlp_hidden, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden, bias=False)
        self.norm = QKNorm(hidden // heads)

    def transform(
        self, values: Tensor, frequencies: Tensor, bounds: Flux3Bounds, modulation: Flux3Modulation
    ) -> Tensor:
        """Return new LD residuals; bound samples never attend across each other."""
        shift, scale, gate = modulation
        normalized = layernorm_modulation(values, scale, shift)
        q = self.q_proj(normalized).reshape(1, values.shape[0], self.heads, -1).transpose(1, 2)
        k = self.k_proj(normalized).reshape(1, values.shape[0], self.heads, -1).transpose(1, 2)
        v = self.v_proj(normalized).reshape(1, values.shape[0], self.heads, -1).transpose(1, 2)
        q, k = self.norm(q, k, frequencies)
        attended = packed_attention(q, k, v.to(q.dtype), bounds)
        return self.feed_forward(values, normalized, attended, gate)

    def feed_forward(self, values: Tensor, normalized: Tensor, attended: Tensor, gate: Tensor) -> Tensor:
        """Recompute parallel FFN/output projections while retaining the attention result."""
        gate_in, up = self.mlp_in(normalized).chunk(2, dim=-1)
        return values + gate * (self.attn_out(attended) + self.mlp_out(F.silu(gate_in) * up))

    forward = transform
