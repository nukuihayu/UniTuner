# Derived from Black Forest Labs FLUX, Apache-2.0.
"""Compute native FLUX joint attention, interleaved rotary pairs and modulation."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from unituner.execution.attention import scaled_dot_product_attention
from unituner.models.components.rotary import apply_rotary


def timestep_embedding(timesteps: Tensor) -> Tensor:
    """Embed floating B noise fractions into Bx256 features using native time*1000.

    Frequencies and trigonometry use FP32; output retains the input dtype.
    """
    frequencies = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128).to(timesteps.device)
    angles = (1000 * timesteps)[:, None].float() * frequencies[None]
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(timesteps)


def rotary_embedding(ids: Tensor, axes: tuple[int, ...], theta: int) -> Tensor:
    """Return Bx1xLx(D/2)x2x2 rotation matrices from per-axis positions.

    Native frequencies and trigonometry are FP64, then matrices are rounded to
    FP32. Position tensors and their gradients remain caller-owned.
    """
    matrices = []
    for index, width in enumerate(axes):
        scale = torch.arange(0, width, 2, dtype=torch.float64, device=ids.device) / width
        angles = torch.einsum("...n,d->...nd", ids[..., index], 1.0 / (theta**scale))
        matrices.append(
            torch.stack((angles.cos(), -angles.sin(), angles.sin(), angles.cos()), dim=-1).reshape(*angles.shape, 2, 2)
        )
    return torch.cat(matrices, dim=-3).float().unsqueeze(1)


def rotate(values: Tensor, positions: Tensor) -> Tensor:
    """Apply FP32 interleaved-pair RoPE and restore BHLD input dtype and shape."""
    return apply_rotary(values, positions)


def attention(q: Tensor, k: Tensor, v: Tensor, positions: Tensor, lengths: tuple[int, ...] | None) -> Tensor:
    """Compute noncausal BHLD attention, returning BxLx(H*D) native outputs.

    Optional host prefix lengths trim both queries and keys. Padded queries are
    zero before the output projection, including heterogeneous Kontext references.
    No RNG, dropout or implicit precision policy is applied.
    """
    q, k = rotate(q, positions), rotate(k, positions)
    if lengths is None:
        result = scaled_dot_product_attention(q, k, v)
    else:
        rows = []
        for index, length in enumerate(lengths):
            row = scaled_dot_product_attention(
                q[index : index + 1, :, :length], k[index : index + 1, :, :length], v[index : index + 1, :, :length]
            )
            rows.append(F.pad(row, (0, 0, 0, q.shape[2] - length)))
        result = torch.cat(rows, dim=0)
    return result.transpose(1, 2).flatten(2)


class MLPEmbedder(nn.Module):
    """Project vector/time features with native SiLU and published parameter keys."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim)

    def project(self, values: Tensor) -> Tensor:
        """Return differentiable Bxhidden_dim features."""
        return self.out_layer(self.silu(self.in_layer(values)))

    forward = project


class RMSNorm(nn.Module):
    """Normalize each head in FP32 with native epsilon and learned scale."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def normalize(self, values: Tensor) -> Tensor:
        """Return input shape/dtype, accumulating squares and scale in FP32."""
        values_fp32 = values.float()
        return (
            (values_fp32 * torch.rsqrt(values_fp32.square().mean(-1, keepdim=True) + 1e-6)) * self.scale.float()
        ).to(values)

    forward = normalize


class QKNorm(nn.Module):
    """Keep independent native query/key normalization parameters."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def normalize(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Normalize Q/K and match V's dtype/device without changing V."""
        return self.query_norm(q).to(v), self.key_norm(k).to(v)

    forward = normalize


class SelfAttention(nn.Module):
    """Own native fused QKV, QK norms and output projection parameters."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.norm = QKNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim)


@dataclass(frozen=True, slots=True)
class ModulationValues:
    """Carry Bx1xD affine and residual-gate tensors without copies."""

    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    """Own one or two native shift/scale/gate groups."""

    def __init__(self, dim: int, *, double: bool) -> None:
        super().__init__()
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim)
        self.activation = nn.SiLU()

    def modulate(self, vector: Tensor) -> tuple[ModulationValues, ModulationValues | None]:
        """Project BxD context and return broadcastable modulation groups."""
        values = self.lin(self.activation(vector))[:, None, :].chunk(self.multiplier, dim=-1)
        return ModulationValues(*values[:3]), ModulationValues(*values[3:]) if self.multiplier == 6 else None

    forward = modulate


def split_qkv(values: Tensor, num_heads: int) -> tuple[Tensor, Tensor, Tensor]:
    """View native fused QKV order as three BHLD tensors."""
    q, k, v = values.reshape(values.shape[0], values.shape[1], 3, num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
    return q, k, v


class DoubleStreamBlock(nn.Module):
    """Update image and text streams with joint attention and separate gated FFNs."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.img_mod = Modulation(dim, double=True)
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim, num_heads, qkv_bias)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(approximate="tanh"), nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.txt_mod = Modulation(dim, double=True)
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim, num_heads, qkv_bias)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(approximate="tanh"), nn.Linear(int(dim * mlp_ratio), dim)
        )

    def transform(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, lengths: tuple[int, ...] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return new image/text BLD tensors; neither input stream is mutated."""
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        if img_mod2 is None or txt_mod2 is None:
            raise RuntimeError("Double-stream blocks require two modulation groups")
        iq, ik, iv = split_qkv(
            self.img_attn.qkv((1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift), self.num_heads
        )
        tq, tk, tv = split_qkv(
            self.txt_attn.qkv((1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift), self.num_heads
        )
        iq, ik = self.img_attn.norm(iq, ik, iv)
        tq, tk = self.txt_attn.norm(tq, tk, tv)
        output = attention(torch.cat((tq, iq), 2), torch.cat((tk, ik), 2), torch.cat((tv, iv), 2), pe, lengths)
        txt_attn, img_attn = output[:, : txt.shape[1]], output[:, txt.shape[1] :]
        img = torch.addcmul(img, img_mod1.gate, self.img_attn.proj(img_attn))
        img = torch.addcmul(
            img, img_mod2.gate, self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        )
        txt = torch.addcmul(txt, txt_mod1.gate, self.txt_attn.proj(txt_attn))
        txt = torch.addcmul(
            txt, txt_mod2.gate, self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        )
        return img, txt

    forward = transform


class SingleStreamBlock(nn.Module):
    """Update joint tokens with fused parallel attention/FFN projections."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.linear1 = nn.Linear(dim, 3 * dim + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(dim + self.mlp_hidden_dim, dim)
        self.norm = QKNorm(dim // num_heads)
        self.pre_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(dim, double=False)

    def transform(self, values: Tensor, vec: Tensor, pe: Tensor, lengths: tuple[int, ...] | None = None) -> Tensor:
        """Return new BLD tokens using native FP32 QK normalization and RoPE."""
        mod, _ = self.modulation(vec)
        qkv, mlp = self.linear1((1 + mod.scale) * self.pre_norm(values) + mod.shift).split(
            (3 * self.hidden_size, self.mlp_hidden_dim), dim=-1
        )
        q, k, v = split_qkv(qkv, self.num_heads)
        q, k = self.norm(q, k, v)
        attended = attention(q, k, v, pe, lengths)
        return self.feed_forward(values, mod.gate, attended, mlp)

    def feed_forward(self, values: Tensor, gate: Tensor, attended: Tensor, mlp: Tensor) -> Tensor:
        """Checkpoint the parallel FFN tail without splitting published fused projection keys."""
        output = self.linear2(torch.cat((attended, self.mlp_act(mlp)), dim=2))
        return torch.addcmul(values, gate, output)

    forward = transform


class LastLayer(nn.Module):
    """Project image tokens after native adaptive final normalization."""

    def __init__(self, dim: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def project(self, values: Tensor, vec: Tensor) -> Tensor:
        """Return BxLxout_channels prediction without changing token layout."""
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        return self.linear((1 + scale[:, None]) * self.norm_final(values) + shift[:, None])

    forward = project
