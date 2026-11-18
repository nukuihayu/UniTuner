# Derived from Krea2 and pinned Musubi Tuner; see THIRD_PARTY_NOTICES.md.
"""Compute Krea2 zero-centered RMSNorm, gated GQA and shared modulation blocks."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unituner.execution.attention import scaled_dot_product_attention
from unituner.models.components.rotary import apply_rotary


def timestep_embedding(timesteps: Tensor, dim: int, *, dtype: torch.dtype) -> Tensor:
    """Return B1D cosine/sine time features using FP32 normalized time*1000."""
    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half)
    angles = (timesteps.float() * 1000)[:, None, None] * frequencies
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)


def rotary_embedding(positions: Tensor, axes: tuple[int, int, int], theta: float) -> Tensor:
    """Return B1L(D/2)22 rotary matrices, computing native frequencies in FP64."""
    matrices = []
    for index, width in enumerate(axes):
        scale = torch.arange(0, width, 2, dtype=torch.float64, device=positions.device) / width
        angles = torch.einsum("...n,d->...nd", positions[..., index], 1.0 / (theta**scale))
        matrices.append(
            torch.stack((angles.cos(), -angles.sin(), angles.sin(), angles.cos()), dim=-1).reshape(*angles.shape, 2, 2)
        )
    return torch.cat(matrices, dim=-3).float()[:, None]


def rotate(values: Tensor, rotary: Tensor) -> Tensor:
    """Rotate native BHLD interleaved pairs in FP32 and restore input dtype."""
    return apply_rotary(values, rotary)


def attention(q: Tensor, k: Tensor, v: Tensor, lengths: tuple[int, ...] | None) -> Tensor:
    """Return BL(H*D) gated-attention inputs with native prefix-padding semantics.

    Expand grouped KV heads before SDPA. Uniform prefixes crop queries/keys then
    zero-pad outputs; heterogeneous lengths mask only keys, retaining padded
    query results. Host lengths avoid device readback inside each block.
    """
    groups = q.shape[1] // k.shape[1]
    if groups != 1:
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
    if lengths is not None and all(length == lengths[0] for length in lengths):
        length = lengths[0]
        result = scaled_dot_product_attention(q[:, :, :length], k[:, :, :length], v[:, :, :length])
        result = F.pad(result, (0, 0, 0, q.shape[2] - length))
    else:
        mask = None
        if lengths is not None:
            mask = torch.arange(k.shape[2], device=q.device)[None, :] < torch.tensor(lengths, device=q.device)[:, None]
            mask = mask[:, None, None]
        result = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return result.transpose(1, 2).flatten(2)


class RMSNorm(nn.Module):
    """Own native zero-initialized scale, applied as scale+1 in FP32 with eps1e-5."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.scale = nn.Parameter(torch.zeros(width))

    def normalize(self, values: Tensor) -> Tensor:
        """Retain native F.rms_norm forward/backward and restore the activation dtype."""
        return F.rms_norm(values.float(), (self.width,), weight=self.scale.float() + 1.0, eps=1e-5).to(values.dtype)

    forward = normalize


class QKNorm(nn.Module):
    """Own independent per-head query and key normalization parameters."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.qnorm = RMSNorm(width)
        self.knorm = RMSNorm(width)


class SwiGLU(nn.Module):
    """Keep separate native gate/up/down projections and round FFN width to128."""

    def __init__(self, width: int, multiplier: int, bias: bool) -> None:
        super().__init__()
        hidden = 128 * ((int(2 * width / 3) * multiplier + 127) // 128)
        self.gate = nn.Linear(width, hidden, bias=bias)
        self.up = nn.Linear(width, hidden, bias=bias)
        self.down = nn.Linear(hidden, width, bias=bias)

    def project(self, values: Tensor) -> Tensor:
        """Return differentiable BL-width gated features in native multiplication order."""
        return self.down(F.silu(self.gate(values)) * self.up(values))

    forward = project


class Attention(nn.Module):
    """Own native query/key/value, sigmoid output gate and output projection."""

    def __init__(self, width: int, heads: int, kv_heads: int, bias: bool) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, width // heads
        self.wq = nn.Linear(width, width, bias=bias)
        self.wk = nn.Linear(width, self.head_dim * kv_heads, bias=bias)
        self.wv = nn.Linear(width, self.head_dim * kv_heads, bias=bias)
        self.gate = nn.Linear(width, width, bias=bias)
        self.qknorm = QKNorm(self.head_dim)
        self.wo = nn.Linear(width, width, bias=bias)

    def attend(self, values: Tensor, rotary: Tensor | None, lengths: tuple[int, ...] | None) -> Tensor:
        """Return BLD native gated attention with optional RoPE and host valid prefixes."""
        batch, count, _ = values.shape
        q = self.wq(values).reshape(batch, count, self.heads, self.head_dim).transpose(1, 2)
        k = self.wk(values).reshape(batch, count, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(values).reshape(batch, count, self.kv_heads, self.head_dim).transpose(1, 2)
        gate = self.gate(values)
        q, k = self.qknorm.qnorm(q), self.qknorm.knorm(k)
        if rotary is not None:
            q, k = rotate(q, rotary), rotate(k, rotary)
        return self.wo(attention(q, k, v, lengths) * torch.sigmoid(gate))

    forward = attend


class TextFusionBlock(nn.Module):
    """Refine either selected-layer tokens or caption tokens with native residual order."""

    def __init__(self, width: int, heads: int, kv_heads: int, multiplier: int, bias: bool) -> None:
        super().__init__()
        self.prenorm, self.postnorm = RMSNorm(width), RMSNorm(width)
        self.attn = Attention(width, heads, kv_heads, bias)
        self.mlp = SwiGLU(width, multiplier, bias)

    def refine(self, values: Tensor, lengths: tuple[int, ...] | None) -> Tensor:
        """Return native prenorm-attention then postnorm-MLP residual features."""
        values = values + self.attn(self.prenorm(values), None, lengths)
        return values + self.mlp(self.postnorm(values))

    forward = refine


class TextFusionTransformer(nn.Module):
    """Fuse selected encoder layers with two layerwise and two token-refiner blocks."""

    def __init__(self, layers: int, width: int, heads: int, kv_heads: int, multiplier: int, bias: bool) -> None:
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [TextFusionBlock(width, heads, kv_heads, multiplier, bias) for _ in range(2)]
        )
        self.projector = nn.Linear(layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [TextFusionBlock(width, heads, kv_heads, multiplier, bias) for _ in range(2)]
        )

    def fuse(self, values: Tensor, lengths: tuple[int, ...]) -> Tensor:
        """Return BLD from BLND, attending over all selected layers before masked text."""
        batch, count, layers, width = values.shape
        values = values.reshape(batch * count, layers, width)
        for block in self.layerwise_blocks:
            values = block(values.contiguous(), None)
        values = self.projector(values.reshape(batch, count, layers, width).permute(0, 1, 3, 2)).squeeze(-1)
        for block in self.refiner_blocks:
            values = block(values, lengths)
        return values

    forward = fuse


class SharedModulation(nn.Module):
    """Own six native zero-initialized offsets for shared time modulation."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * width))

    def modulate(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return prescale/preshift/pregate/postscale/postshift/postgate in native order."""
        prescale, preshift, pregate, postscale, postshift, postgate = (values + self.lin).chunk(6, dim=-1)
        return prescale, preshift, pregate, postscale, postshift, postgate

    forward = modulate


class SingleStreamBlock(nn.Module):
    """Apply shared time modulation to image-first joint attention and SwiGLU."""

    def __init__(self, width: int, heads: int, kv_heads: int, multiplier: int, bias: bool) -> None:
        super().__init__()
        self.mod = SharedModulation(width)
        self.prenorm, self.postnorm = RMSNorm(width), RMSNorm(width)
        self.attn = Attention(width, heads, kv_heads, bias)
        self.mlp = SwiGLU(width, multiplier, bias)

    def refine(self, values: Tensor, modulation: Tensor, rotary: Tensor, lengths: tuple[int, ...]) -> Tensor:
        """Return joint sequence, preserving attention then FFN residual arithmetic."""
        scale, shift, gate, postscale, postshift, postgate = self.mod(modulation)
        values = values + gate * self.attn((1 + scale) * self.prenorm(values) + shift, rotary, lengths)
        return values + postgate * self.mlp((1 + postscale) * self.postnorm(values) + postshift)

    forward = refine


class FinalModulation(nn.Module):
    """Own native two-row offsets for final time scale/shift."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, width))

    def modulate(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Return B1D scale and shift using native broadcasting over the two offsets."""
        scale, shift = (values + self.lin[None]).chunk(2, dim=1)
        return scale, shift

    forward = modulate


class LastLayer(nn.Module):
    """Normalize/modulate the final sequence and project native pixel-patch channels."""

    def __init__(self, width: int, patch: int, channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(width)
        self.linear = nn.Linear(width, patch * patch * channels)
        self.modulation = FinalModulation(width)

    def project(self, values: Tensor, time_features: Tensor) -> Tensor:
        """Return native patch predictions before selecting leading image tokens."""
        scale, shift = self.modulation(time_features)
        return self.linear((1 + scale) * self.norm(values) + shift)

    forward = project
