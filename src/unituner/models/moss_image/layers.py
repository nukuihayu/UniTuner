# Adapted from OpenMOSS MOSS-Image and kohya-ss/sd-scripts, Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""MOSS RMS normalization, SiTU-GLU, four-axis rotary encoding and residual blocks."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unituner.execution.attention import scaled_dot_product_attention


class RMSNorm(nn.Module):
    """Normalize in FP32, round to activation dtype, then apply the learned weight."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def normalize(self, value: Tensor) -> Tensor:
        """Normalize the last dimension in FP32 and apply learned dtype-preserving scale."""
        # Keep the two cast branches from the published pipeline: merging them
        # changes BF16 input-gradient rounding even with identical forward output.
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(value.dtype) * self.weight

    forward = normalize


class SiTUGLU(nn.Module):
    """Bounded SiLU gate and bounded hidden projection, preserving native BF16 VJP."""

    def activate(self, hidden: Tensor, gate: Tensor) -> Tensor:
        """Return tanh-bounded SiLU gating with unchanged input gradient paths."""
        return (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (25 * torch.tanh(hidden / 25))

    forward = activate


def apply_rope(value: Tensor, rope: Tensor) -> Tensor:
    """Rotate BLHD using BLD sin/cos and half-split feature pairs, not adjacent pairs."""
    sine, cosine = rope.chunk(2, -1)
    sine = torch.cat((sine, sine), -1).unsqueeze(-2)
    cosine = torch.cat((cosine, cosine), -1).unsqueeze(-2)
    first, second = value.chunk(2, -1)
    return value * cosine + torch.cat((-second, first), -1) * sine


def rotary_attention(query: Tensor, key: Tensor, value: Tensor, rope: Tensor, lengths: tuple[int, ...]) -> Tensor:
    """Rotate BLHD Q/K and attend only valid prefixes, returning zero-padded BLD.

    Use Torch SDPA without dropout, keeping native half-split rotary pairing and
    Q/K casting to the value dtype. Equal lengths use a single batched call;
    unequal lengths crop each row. Pad before the published output projection.
    """
    query, key = apply_rope(query, rope).to(value.dtype), apply_rope(key, rope).to(value.dtype)
    maximum = value.shape[1]
    if all(length == lengths[0] for length in lengths):
        size = lengths[0]
        output = scaled_dot_product_attention(
            query[:, :size].transpose(1, 2), key[:, :size].transpose(1, 2), value[:, :size].transpose(1, 2)
        ).transpose(1, 2)
        return F.pad(output, (0, 0, 0, 0, 0, maximum - size)).flatten(2)
    rows = []
    for index, size in enumerate(lengths):
        output = scaled_dot_product_attention(
            query[index : index + 1, :size].transpose(1, 2),
            key[index : index + 1, :size].transpose(1, 2),
            value[index : index + 1, :size].transpose(1, 2),
        ).transpose(1, 2)
        rows.append(F.pad(output, (0, 0, 0, 0, 0, maximum - size)))
    return torch.cat(rows).flatten(2)


class RotaryEmbedding(nn.Module):
    """Derived FP32 frequencies for (reference, frame/text, centered height, width)."""

    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self.default_bands(torch.device("cpu")))

    def default_bands(self, device: torch.device) -> Tensor:
        """Construct on CPU in FP32, independent of checkpoint or model precision."""
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / self.base**exponent).to(device)

    def embed(self, height: int, width: int, text_length: int, lengths: Tensor) -> Tensor:
        """Return [B,H*W+L,head_dim] sine/cosine for centered image and prefix text coordinates."""
        device = lengths.device
        # Module.to() can round a persistent frequency buffer; never reuse that value.
        bands = self.default_bands(device)
        rows, columns = torch.meshgrid(
            torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
        )
        image = torch.stack((torch.zeros_like(rows), torch.zeros_like(rows), rows, columns), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        image = image[None].repeat(lengths.shape[0], 1, 1)
        image[:, :, 1] = lengths.float()[:, None]
        text = torch.zeros(text_length, 4, device=device)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text[None].expand(lengths.shape[0], -1, -1)), 1).unsqueeze(-1) * bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)

    forward = embed


class TextBlock(nn.Module):
    """Unmodulated caption refiner with biased projections and fused QKV."""

    def __init__(self, dim: int, ffn_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up, self.ffn_down = (
            nn.Linear(dim, ffn_dim),
            nn.Linear(dim, ffn_dim),
            nn.Linear(ffn_dim, dim),
        )
        self.activation = SiTUGLU()

    def feed_forward(self, value: Tensor) -> Tensor:
        """Apply norm and biased SiTU-GLU projections, preserving residual ownership."""
        normalized = self.norm2(value)
        return self.ffn_down(self.activation(self.ffn_up(normalized), self.ffn_gate(normalized)))

    def refine(self, value: Tensor, rope: Tensor, lengths: tuple[int, ...]) -> Tensor:
        """Refine BLD captions with prefix attention and an unmodulated residual FFN."""
        query, key, projected = self.qkv(self.norm1(value)).chunk(3, -1)
        query = self.q_norm(query.unflatten(-1, (self.heads, -1)))
        key = self.k_norm(key.unflatten(-1, (self.heads, -1)))
        value = value + self.proj(
            rotary_attention(query, key, projected.unflatten(-1, (self.heads, -1)), rope, lengths)
        )
        residual = self.feed_forward(value)
        return value + residual

    forward = refine


class TextEmbedder(nn.Module):
    """Own the published text projection and unmodulated refinement blocks."""

    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int, count: int) -> None:
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(count)])


class BlockProjections(nn.Module):
    """Concrete published projection names, including hidden-first fused FFN."""

    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim, bias=False) for _ in range(4))
        self.ffn_up_gate = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, dim, bias=False)


class MossImageBlock(nn.Module):
    """Joint image/text residual block with shared timestep modulation."""

    def __init__(self, dim: int, ffn_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = BlockProjections(dim, ffn_dim)
        self.activation = SiTUGLU()

    def feed_forward(self, value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        """Modulate normalized BLD features and split fused FFN as hidden then gate."""
        normalized = self.norm2(value) * (1 + scale[:, None]) + shift[:, None]
        hidden, gate = self.proj.ffn_up_gate(normalized.to(self.norm2.weight.dtype)).chunk(2, -1)
        return self.proj.ffn_down(self.activation(hidden, gate))

    def attention_inputs(self, value: Tensor, rope: Tensor, modulation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project Q/K/V with native normalization and already-applied rotary positions."""
        shift, scale = modulation[:, 0], modulation[:, 1]
        normalized = (self.norm1(value) * (1 + scale[:, None]) + shift[:, None]).to(self.norm1.weight.dtype)
        query = self.q_norm(self.proj.q(normalized).unflatten(-1, (self.heads, -1)))
        key = self.k_norm(self.proj.k(normalized).unflatten(-1, (self.heads, -1)))
        projected = self.proj.v(normalized).unflatten(-1, (self.heads, -1))
        return apply_rope(query, rope).to(projected.dtype), apply_rope(key, rope).to(projected.dtype), projected

    def finish(self, value: Tensor, attended: Tensor, modulation: Tensor) -> Tensor:
        """Apply the native attention and FFN residuals for one time-conditioned stream."""
        value = value + self.proj.o(attended) * modulation[:, 2, None]
        return value + self.feed_forward(value, modulation[:, 3], modulation[:, 4]) * modulation[:, 5, None]

    def transform(
        self,
        value: Tensor,
        rope: Tensor,
        modulation: Tensor,
        lengths: tuple[int, ...],
        reference_count: int = 0,
        reference_modulation: Tensor | None = None,
        reference_kv: tuple[Tensor, Tensor] | None = None,
        kv_output: list[tuple[Tensor, Tensor]] | None = None,
    ) -> Tensor:
        """Transform body prefixes and optional isolated zero-time reference suffixes.

        Reference K/V remain differentiable during training. Cached K/V are supplied
        only by the sampler, scoped to one fixed-weight inference call.
        """
        if reference_count or reference_kv is not None:
            from unituner.models.moss_image.editing import attend_body

            body_length = value.shape[1] - reference_count
            body, body_rope = value[:, :body_length], rope[:, :body_length]
            query, key, projected = self.attention_inputs(body, body_rope, modulation)
            reference_output = None
            if reference_count:
                if reference_modulation is None or reference_kv is not None:
                    raise ValueError("Live references require zero-time modulation and no cached K/V")
                reference = value[:, body_length:]
                rq, rk, rv = self.attention_inputs(reference, rope[:, body_length:], reference_modulation)
                reference_kv = (rk, rv)
                attended = scaled_dot_product_attention(rq.transpose(1, 2), rk.transpose(1, 2), rv.transpose(1, 2))
                reference_output = self.finish(reference, attended.transpose(1, 2).flatten(2), reference_modulation)
            if reference_kv is None:
                raise ValueError("Missing reference K/V")
            body = self.finish(body, attend_body(query, key, projected, lengths, reference_kv), modulation)
            return body if reference_output is None else torch.cat((body, reference_output), 1)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        normalized = (self.norm1(value) * (1 + scale_a[:, None]) + shift_a[:, None]).to(self.norm1.weight.dtype)
        query = self.q_norm(self.proj.q(normalized).unflatten(-1, (self.heads, -1)))
        key = self.k_norm(self.proj.k(normalized).unflatten(-1, (self.heads, -1)))
        projected = self.proj.v(normalized).unflatten(-1, (self.heads, -1))
        if kv_output is not None:
            if torch.is_grad_enabled() or self.training:
                raise ValueError("Reference cache capture is inference-only")
            kv_output.append((apply_rope(key, rope).to(projected.dtype), projected))
        value = value + self.proj.o(rotary_attention(query, key, projected, rope, lengths)) * gate_a[:, None]
        residual = self.feed_forward(value, shift_f, scale_f)
        return value + residual * gate_f[:, None]

    forward = transform
