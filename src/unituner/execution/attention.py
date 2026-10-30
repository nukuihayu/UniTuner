"""Select explicit inference attention while retaining Torch's differentiable training path."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from importlib.util import find_spec
from threading import local
from typing import Literal, Protocol, cast

import torch
from torch import Tensor
from torch.nn import functional as F

from unituner.execution.activations import AttentionActivations, current_attention_activations, offloaded_attention

type AttentionBackend = Literal["torch", "kitchen-int8"]


class _KitchenAttention(Protocol):
    def __call__(self, q: Tensor, k: Tensor, v: Tensor, *, scale: float | None, attn_mask: Tensor | None) -> Tensor: ...


_backend: ContextVar[_KitchenAttention | None] = ContextVar("unituner_inference_attention", default=None)


class _CompiledAttention(local):
    """Expose an invocation's resolved context through Dynamo-guarded thread-local fields."""

    def __init__(self) -> None:
        self.active = False
        self.fullgraph = False
        self.provider: _KitchenAttention | None = None
        self.activations: AttentionActivations | None = None


_compiled = _CompiledAttention()


@contextmanager
def compiled_attention(*, fullgraph: bool) -> Generator[None]:
    """Snapshot context outside tracing; restore nested/thread-local state even on failure.

    Dynamo guards these fields, so a later invocation with a different offload
    owner cannot reuse a graph that silently ignores the requested policy.
    ContextVars remain the source of truth for inference and checkpoint replay.
    """
    provider, activations = _backend.get(), current_attention_activations()
    if provider is not None and torch.is_grad_enabled():
        raise ValueError("Kitchen INT8 attention is inference-only; disable gradients explicitly")
    previous = _compiled.active, _compiled.fullgraph, _compiled.provider, _compiled.activations
    _compiled.active, _compiled.fullgraph = True, fullgraph
    _compiled.provider, _compiled.activations = provider, activations
    try:
        yield
    finally:
        _compiled.active, _compiled.fullgraph, _compiled.provider, _compiled.activations = previous


@contextmanager
def inference_attention(backend: AttentionBackend) -> Generator[None]:
    """Scope an inference-only provider to this context, restoring it even after failure.

    Kitchen is an optional dependency loaded only at this provider boundary.
    Torch is the default; explicit unsupported Kitchen calls never fall back.
    """
    provider = None
    if backend == "kitchen-int8":
        if find_spec("comfy_kitchen") is None:
            raise ValueError("Kitchen inference attention requires the kitchen extra: uv sync --extra kitchen")
        module = import_module("comfy_kitchen")
        provider = cast(_KitchenAttention, module.int8_attention)
    elif backend != "torch":
        raise ValueError(f"Unsupported inference attention backend: {backend}")
    token = _backend.set(provider)
    try:
        yield
    finally:
        _backend.reset(token)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """Compute BHLD SDPA, preserving Torch gradients unless explicit inference selects INT8.

    Kitchen's Q/K/V must be same-device CUDA, matching floating dtypes and head
    widths 1..256 with contiguous final dimension. Quantization changes outputs.
    Constant broadcast masks and GQA are supported. Causal masks must be explicit;
    dropout and gradient-enabled use are rejected by the inference provider.

    AMP and low-precision Q/K/V preserve native SDPA rounding across offload policies.
    Explicit fullgraph owners trace the resolved, guarded context without a
    graph break; FP32 retains compiled native arithmetic.
    """
    if torch.compiler.is_compiling() and _compiled.active and _compiled.fullgraph:
        return _attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)
    if torch.is_autocast_enabled(q.device.type) or q.dtype in (torch.float16, torch.bfloat16):
        return _native_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)
    return _attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)


@torch.compiler.disable()
def _native_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    dropout_p: float,
    is_causal: bool,
    *,
    scale: float | None,
    enable_gqa: bool,
) -> Tensor:
    """Preserve native SDPA arithmetic across compiler and activation-offload boundaries."""
    return _attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)


def _attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    dropout_p: float,
    is_causal: bool,
    *,
    scale: float | None,
    enable_gqa: bool,
) -> Tensor:
    """Dispatch native differentiable SDPA or the explicitly selected inference provider."""
    if torch.compiler.is_compiling() and _compiled.active:
        provider, activations = _compiled.provider, _compiled.activations
    else:
        provider, activations = _backend.get(), current_attention_activations()
    if provider is None:
        if activations is not None and torch.is_grad_enabled():
            return offloaded_attention(activations, q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa
        )
    if torch.is_grad_enabled():
        raise ValueError("Kitchen INT8 attention is inference-only; disable gradients explicitly")
    if dropout_p or is_causal:
        raise ValueError("Kitchen INT8 inference requires zero dropout and an explicit causal mask")
    if not enable_gqa and k.shape[1] not in (1, q.shape[1]):
        raise ValueError("Unequal attention head counts require enable_gqa=True")
    return provider(q, k, v, scale=scale, attn_mask=attn_mask)
