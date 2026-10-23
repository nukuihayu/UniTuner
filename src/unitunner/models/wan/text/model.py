# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Compute native Wan UMT5 features with unscaled attention and signed relative buckets."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unitunner.models.wan.config.t5 import WanT5Config


class T5Norm(nn.Module):
    """Preserve native RMS reduction and post-reduction scale multiplication."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def normalize(self, values: Tensor) -> Tensor:
        """Return native FP32 RMS-normalized BLD activations with epsilon1e-6."""
        values = values * torch.rsqrt(values.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return self.weight * values.to(self.weight.dtype)

    forward = normalize


class RelativeEmbedding(nn.Module):
    """Own bidirectional signed T5 buckets with a fixed native maximum distance128."""

    def __init__(self, buckets: int, heads: int) -> None:
        super().__init__()
        self.num_buckets = buckets
        self.embedding = nn.Embedding(buckets, heads)

    def embed(self, length: int) -> Tensor:
        """Return contiguous 1HLL relative bias using native FP32 logarithmic bins."""
        positions = torch.arange(length, device=self.embedding.weight.device)
        relative = positions.unsqueeze(0) - positions.unsqueeze(1)
        buckets = self.num_buckets // 2
        signs = (relative > 0).long() * buckets
        distance = relative.abs()
        exact = buckets // 2
        logarithmic = exact + (torch.log(distance.float() / exact) / math.log(128 / exact) * (buckets - exact)).long()
        logarithmic = torch.minimum(logarithmic, torch.full_like(logarithmic, buckets - 1))
        indices = signs + torch.where(distance < exact, distance, logarithmic)
        return self.embedding(indices).permute(2, 0, 1).unsqueeze(0).contiguous()

    forward = embed


class T5Attention(nn.Module):
    """Own native bias-free Q/K/V/O projections; attention has no head-width scaling."""

    def __init__(self, config: WanT5Config) -> None:
        super().__init__()
        self.num_heads, self.head_dim = config.num_heads, config.attention_dim // config.num_heads
        self.q = nn.Linear(config.hidden_size, config.attention_dim, bias=False)
        self.k = nn.Linear(config.hidden_size, config.attention_dim, bias=False)
        self.v = nn.Linear(config.hidden_size, config.attention_dim, bias=False)
        self.o = nn.Linear(config.attention_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def attend(self, values: Tensor, mask: Tensor | None, position_bias: Tensor) -> Tensor:
        """Return BLD using FP32-softmax native einsums and additive minimum-finite masking."""
        batch, length, _ = values.shape
        heads, width = self.num_heads, self.head_dim
        q = self.q(values).view(batch, length, heads, width)
        k = self.k(values).view(batch, length, heads, width)
        v = self.v(values).view(batch, length, heads, width)
        bias = values.new_zeros(batch, heads, length, length) + position_bias
        if mask is not None:
            visibility = mask.view(batch, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            bias = bias.masked_fill(visibility == 0, torch.finfo(values.dtype).min)
        scores = torch.einsum("binc,bjnc->bnij", q, k) + bias
        probabilities = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        result = torch.einsum("bnij,bjnc->binc", probabilities, v)
        return self.dropout(self.o(result.reshape(batch, length, heads * width)))

    forward = attend


class T5GELU(nn.Module):
    """Compute the native explicit tanh approximation, preserving expression order."""

    def activate(self, values: Tensor) -> Tensor:
        """Return differentiable tanh GELU without substituting a fused operator."""
        return (
            0.5 * values * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (values + 0.044715 * torch.pow(values, 3.0))))
        )

    forward = activate


class T5FeedForward(nn.Module):
    """Own native GELU gate/up/down paths and both dropout invocation sites."""

    def __init__(self, config: WanT5Config) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(config.hidden_size, config.ffn_dim, bias=False), T5GELU())
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def project(self, values: Tensor) -> Tensor:
        """Preserve up-times-gate multiplication before dropout and output projection."""
        values = self.dropout(self.fc1(values) * self.gate(values))
        return self.dropout(self.fc2(values))

    forward = project


class T5Block(nn.Module):
    """Apply native prenorm attention and gated feed-forward residuals."""

    def __init__(self, config: WanT5Config) -> None:
        super().__init__()
        self.norm1 = T5Norm(config.hidden_size)
        self.attn = T5Attention(config)
        self.norm2 = T5Norm(config.hidden_size)
        self.ffn = T5FeedForward(config)
        self.pos_embedding = (
            None if config.shared_position_bias else RelativeEmbedding(config.num_buckets, config.num_heads)
        )

    def refine(self, values: Tensor, mask: Tensor | None, position_bias: Tensor | None) -> Tensor:
        """Return BLD using the owned per-block or supplied shared relative bias."""
        bias = position_bias if self.pos_embedding is None else self.pos_embedding(values.shape[1])
        if bias is None:
            raise ValueError("Wan T5 block requires relative position bias")
        values = values + self.attn(self.norm1(values), mask, bias)
        return values + self.ffn(self.norm2(values))

    forward = refine


class WanT5Encoder(nn.Module):
    """Own native Wan UMT5 encoder keys and full padded-sequence computation.

    Execution supports FP32/FP16/BF16 activations without autocast, with optional frozen packed projections.
    The text owner crops valid prefixes after the final norm. Integer IDs are
    not differentiable; every active model parameter retains its gradient.
    """

    def __init__(self, config: WanT5Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = (
            RelativeEmbedding(config.num_buckets, config.num_heads) if config.shared_position_bias else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([T5Block(config) for _ in range(config.num_layers)])
        self.norm = T5Norm(config.hidden_size)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Retain native UMT5 initialization scales and recursive traversal order."""
        config = self.config
        for module in self.modules():
            if isinstance(module, T5Norm):
                nn.init.ones_(module.weight)
            elif isinstance(module, T5Attention):
                nn.init.normal_(module.q.weight, std=(config.hidden_size * config.attention_dim) ** -0.5)
                nn.init.normal_(module.k.weight, std=config.hidden_size**-0.5)
                nn.init.normal_(module.v.weight, std=config.hidden_size**-0.5)
                nn.init.normal_(module.o.weight, std=(config.num_heads * config.attention_dim) ** -0.5)
            elif isinstance(module, T5FeedForward):
                gate = module.gate[0]
                if not isinstance(gate, nn.Linear):
                    raise TypeError("Wan T5 gate must own a linear projection")
                nn.init.normal_(gate.weight, std=config.hidden_size**-0.5)
                nn.init.normal_(module.fc1.weight, std=config.hidden_size**-0.5)
                nn.init.normal_(module.fc2.weight, std=config.ffn_dim**-0.5)
            elif isinstance(module, RelativeEmbedding):
                nn.init.normal_(module.embedding.weight, std=(2 * config.num_buckets * config.num_heads) ** -0.5)

    def encode(self, ids: Tensor, mask: Tensor | None = None) -> Tensor:
        """Encode nonempty BL int64 IDs with optional BL/BLL key visibility.

        Masks follow native minimum-finite additive bias, including padded
        query outputs. The caller validates ID range and actual mask values on
        the host before device transfer. No RCM input or normalization is used.
        """
        parameter = self.token_embedding.weight
        if ids.ndim != 2 or min(ids.shape) < 1 or ids.dtype != torch.int64 or ids.device != parameter.device:
            raise ValueError("Wan T5 requires nonempty BL int64 IDs on the model device")
        if parameter.dtype not in (torch.float32, torch.float16, torch.bfloat16) or torch.is_autocast_enabled(
            parameter.device.type
        ):
            raise ValueError("Wan T5 requires FP32/FP16/BF16 parameters with autocast disabled")
        if mask is not None and (
            mask.shape not in (ids.shape, (*ids.shape, ids.shape[1]))
            or mask.device != ids.device
            or mask.dtype not in (torch.bool, torch.int64)
        ):
            raise ValueError("Wan T5 visibility must be bool/int64 BL or BLL on the model device")
        values = self.dropout(self.token_embedding(ids))
        bias = None if self.pos_embedding is None else self.pos_embedding(ids.shape[1])
        for block in self.blocks:
            values = block(values, mask, bias)
        return self.dropout(self.norm(values))

    forward = encode
