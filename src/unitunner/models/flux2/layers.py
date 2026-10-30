# Derived from Black Forest Labs FLUX, Apache-2.0.
"""Compute FLUX.2 bias-free SwiGLU streams and globally shared modulation."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from unitunner.execution.attention import scaled_dot_product_attention
from unitunner.models.flux.layers import rotary_embedding as rotary_embedding
from unitunner.models.flux.layers import rotate, split_qkv

type ModulationValues = tuple[Tensor, Tensor, Tensor]
type DoubleModulation = tuple[ModulationValues, ModulationValues]


def timestep_embedding(timesteps: Tensor) -> Tensor:
    """Embed B noise fractions as Bx256 features with native device-local FP32 math.

    Scale time by1000 before FP32 trigonometry and return the input dtype.
    Frequencies are evaluated on the time device, preserving CUDA rounding and
    guidance derivatives instead of transferring host-evaluated frequencies.
    """
    frequencies = torch.exp(-math.log(10000) * torch.arange(128, device=timesteps.device, dtype=torch.float32) / 128)
    angles = (1000 * timesteps)[:, None].float() * frequencies[None]
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(timesteps)


def attention(q: Tensor, k: Tensor, v: Tensor, positions: Tensor) -> Tensor:
    """Apply interleaved FP32 RoPE and unmasked BHLD SDPA, returning BLD tokens."""
    result = scaled_dot_product_attention(rotate(q, positions), rotate(k, positions), v)
    return result.transpose(1, 2).flatten(2)


class RMSNorm(nn.Module):
    """Normalize each head in FP32, then round before learned-scale multiplication."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def normalize(self, values: Tensor) -> Tensor:
        """Preserve native low-precision operation order and parameter keys."""
        fp32 = values.float()
        return (fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + 1e-6)).to(values.dtype) * self.scale

    forward = normalize


class QKNorm(nn.Module):
    """Own independent query/key head normalization parameters."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def normalize(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Return Q/K in V's dtype and device without changing V."""
        return self.query_norm(q).to(v), self.key_norm(k).to(v)

    forward = normalize


class SelfAttention(nn.Module):
    """Own published bias-free fused QKV, QK norms and output projection."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = QKNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim, bias=False)


class SwiGLU(nn.Module):
    """Activate the first half of fused gate/value features with SiLU."""

    def activate(self, values: Tensor) -> Tensor:
        """Return BxLxD from BxLx2D without mutating the fused input."""
        gate, value = values.chunk(2, dim=-1)
        return F.silu(gate) * value

    forward = activate


class MLPEmbedder(nn.Module):
    """Own native bias-free timestep/guidance projections."""

    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(in_dim, dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(dim, dim, bias=False)

    def project(self, values: Tensor) -> Tensor:
        """Return differentiable BxD embeddings in caller precision."""
        return self.out_layer(self.silu(self.in_layer(values)))

    forward = project


class Modulation(nn.Module):
    """Compute globally shared native shift/scale/gate groups in FP32."""

    def __init__(self, dim: int, *, double: bool) -> None:
        super().__init__()
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=False)
        self.activation = nn.SiLU()

    def modulate(self, vector: Tensor) -> tuple[ModulationValues, ModulationValues | None]:
        """Return broadcastable Bx1xD groups rounded to the input vector dtype.

        Native modulation weights must remain FP32 outside autocast. Explicit
        low-precision execution requires its own caller precision policy.
        """
        values = self.lin(self.activation(vector.float()))[:, None].to(vector.dtype).chunk(self.multiplier, dim=-1)
        first = (values[0], values[1], values[2])
        second = (values[3], values[4], values[5]) if self.multiplier == 6 else None
        return first, second

    forward = modulate


class DoubleStreamBlock(nn.Module):
    """Update separate image/text streams with joint attention and SwiGLU FFNs."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        width = int(dim * mlp_ratio)
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim, num_heads)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(nn.Linear(dim, 2 * width, bias=False), SwiGLU(), nn.Linear(width, dim, bias=False))
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim, num_heads)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(nn.Linear(dim, 2 * width, bias=False), SwiGLU(), nn.Linear(width, dim, bias=False))

    def transform(
        self, img: Tensor, txt: Tensor, positions: Tensor, mod_img: DoubleModulation, mod_txt: DoubleModulation
    ) -> tuple[Tensor, Tensor]:
        """Return new BLD streams using shared per-sample modulation, without padding."""
        (is1, ic1, ig1), (is2, ic2, ig2) = mod_img
        (ts1, tc1, tg1), (ts2, tc2, tg2) = mod_txt
        iq, ik, iv = split_qkv(self.img_attn.qkv((1 + ic1) * self.img_norm1(img) + is1), self.num_heads)
        tq, tk, tv = split_qkv(self.txt_attn.qkv((1 + tc1) * self.txt_norm1(txt) + ts1), self.num_heads)
        iq, ik = self.img_attn.norm(iq, ik, iv)
        tq, tk = self.txt_attn.norm(tq, tk, tv)
        output = attention(torch.cat((tq, iq), 2), torch.cat((tk, ik), 2), torch.cat((tv, iv), 2), positions)
        txt_attn, img_attn = output[:, : txt.shape[1]], output[:, txt.shape[1] :]
        img = img + ig1 * self.img_attn.proj(img_attn)
        img = img + ig2 * self.img_mlp((1 + ic2) * self.img_norm2(img) + is2)
        txt = txt + tg1 * self.txt_attn.proj(txt_attn)
        txt = txt + tg2 * self.txt_mlp((1 + tc2) * self.txt_norm2(txt) + ts2)
        return img, txt

    forward = transform


class SingleStreamBlock(nn.Module):
    """Update joint text/image tokens with fused parallel attention and SwiGLU."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.linear1 = nn.Linear(dim, dim * 3 + self.mlp_hidden_dim * 2, bias=False)
        self.linear2 = nn.Linear(dim + self.mlp_hidden_dim, dim, bias=False)
        self.norm = QKNorm(dim // num_heads)
        self.pre_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SwiGLU()

    def transform(self, values: Tensor, positions: Tensor, mod: ModulationValues) -> Tensor:
        """Return new BLD tokens; preserve native attention/FFN projection order."""
        shift, scale, gate = mod
        qkv, mlp = self.linear1((1 + scale) * self.pre_norm(values) + shift).split(
            (3 * self.hidden_size, 2 * self.mlp_hidden_dim), dim=-1
        )
        q, k, v = split_qkv(qkv, self.num_heads)
        q, k = self.norm(q, k, v)
        attended = attention(q, k, v, positions)
        return self.feed_forward(values, gate, attended, mlp)

    def feed_forward(self, values: Tensor, gate: Tensor, attended: Tensor, mlp: Tensor) -> Tensor:
        """Keep shared QKV/FFN input projection resident and recompute the parallel FFN tail."""
        return values + gate * self.linear2(torch.cat((attended, self.mlp_act(mlp)), dim=2))

    forward = transform


class LastLayer(nn.Module):
    """Project native image tokens after FP32 adaptive final normalization."""

    def __init__(self, dim: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=False))

    def project(self, values: Tensor, vector: Tensor) -> Tensor:
        """Return BxLxC velocity rounded to stream dtype; weights follow FP32 policy."""
        shift, scale = self.adaLN_modulation(vector.float()).chunk(2, dim=-1)
        result = self.linear((1 + scale[:, None]) * self.norm_final(values.float()) + shift[:, None])
        return result.to(values.dtype)

    forward = project
