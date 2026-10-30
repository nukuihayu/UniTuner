"""Offload SDPA's saved activations without transferring frozen weights or changing strides."""

from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from types import TracebackType

import torch
from torch import Tensor
from torch.nn import functional as F

from unituner.execution.regions import ModelRegion
from unituner.models.components.forward_override import ForwardOverride


@dataclass(frozen=True, slots=True)
class _SavedActivation:
    host: Tensor
    device: torch.device
    shape: torch.Size
    stride: tuple[int, ...]
    ready: torch.Event


_attention_replay: ContextVar[bool] = ContextVar("attention_replay", default=False)


class AttentionReplay:
    """Mark reusable checkpoint replay scopes so short-lived saved tensors stay on CUDA."""

    def __init__(self) -> None:
        self._tokens: list[Token[bool]] = []

    def __enter__(self) -> None:
        self._tokens.append(_attention_replay.set(True))

    def __exit__(
        self, error_type: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        _attention_replay.reset(self._tokens.pop())


class AttentionActivations:
    """Own an independent D2H stream; autograd packets own pinned storage until graph release.

    Save only non-leaf CUDA activations inside SDPA. Preserve storage span and
    native strides, including sliced and broadcast views. Demand H2D uses the
    backward compute stream and waits on the corresponding D2H event. There is
    no quantization, persistent GPU activation cache or frozen-weight transfer.
    """

    def __init__(self, device: torch.device) -> None:
        self._stream: torch.cuda.Stream = torch.cuda.Stream(device=device)
        self.offloaded_bytes: int = 0
        self.restored_bytes: int = 0
        self._closed: bool = False

    def pack(self, tensor: Tensor) -> Tensor | _SavedActivation:
        """Copy a saved non-leaf activation's storage span, preserving its logical view."""
        if _attention_replay.get():
            # Returning the saved tensor itself retains its grad_fn and creates
            # a cycle through checkpoint replay. Only its storage is needed.
            return tensor.detach()
        if tensor.device.type != "cuda" or tensor.is_leaf or tensor.layout != torch.strided or not tensor.numel():
            return tensor.detach()
        if self._closed:
            raise RuntimeError("Attention activation offload is closed")
        span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
        source = tensor.detach().as_strided((span,), (1,), tensor.storage_offset())
        host = torch.empty(span, dtype=tensor.dtype, device="cpu", pin_memory=True)
        self._stream.wait_stream(torch.cuda.current_stream(tensor.device))
        with torch.cuda.stream(self._stream):
            source.record_stream(self._stream)
            host.copy_(source, non_blocking=True)
            ready = self._stream.record_event()
        self.offloaded_bytes += host.nbytes
        return _SavedActivation(host, tensor.device, tensor.shape, tensor.stride(), ready)

    def unpack(self, packet: Tensor | _SavedActivation) -> Tensor:
        """Restore fresh storage for each read, including retain_graph and repeated backward."""
        if isinstance(packet, Tensor):
            return packet
        if self._closed:
            raise RuntimeError("Attention activation offload closed before backward finished")
        torch.cuda.current_stream(packet.device).wait_event(packet.ready)
        restored = packet.host.to(packet.device, non_blocking=True)
        self.restored_bytes += packet.host.nbytes
        return restored.as_strided(packet.shape, packet.stride)

    def close(self) -> None:
        """Complete D2H work before releasing this invocation policy."""
        if not self._closed:
            self._stream.synchronize()
            self._closed = True


_attention_activations: ContextVar[AttentionActivations | None] = ContextVar("attention_activations", default=None)


def current_attention_activations() -> AttentionActivations | None:
    """Resolve the explicit training scope; ordinary training and preview have no offload hooks."""
    return _attention_activations.get()


@torch.compiler.disable()
def offloaded_attention(
    owner: AttentionActivations,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> Tensor:
    """Keep saved-tensor hooks outside Dynamo while compiling surrounding block computation."""
    with torch.autograd.graph.saved_tensors_hooks(owner.pack, owner.unpack):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa
        )


class AttentionOffloadInstallation[R: Tensor | tuple[Tensor, ...], **P]:
    """Scope one region's forward to an owned SDPA saved-activation policy."""

    def __init__(self, region: ModelRegion, owner: AttentionActivations) -> None:
        self.region: ModelRegion = region
        original: Callable[P, R] = region.module.forward

        def installed(*args: P.args, **kwargs: P.kwargs) -> R:
            if not region.module.training or not torch.is_grad_enabled():
                return original(*args, **kwargs)
            token = _attention_activations.set(owner)
            try:
                return original(*args, **kwargs)
            finally:
                _attention_activations.reset(token)

        self._forward = ForwardOverride(region.module, installed, owner="AttentionOffloadInstallation")

    def close(self) -> None:
        """Restore the preceding forward after all backward reads finish."""
        self._forward.close()
