"""Stage immutable CPU block weights for one CUDA invocation and its backward replay."""

from collections.abc import Callable
from contextlib import nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from types import TracebackType

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint
from torch.utils.hooks import RemovableHandle

from unituner.execution.activations import AttentionReplay
from unituner.execution.mapped_weights import VbarArena, VbarStatistics, VbarWeights
from unituner.execution.prefetch import WeightPrefetch
from unituner.execution.regions import ModelRegion, ModelRegions
from unituner.execution.weight_cache import H2dCache, H2dWeights
from unituner.execution.weight_lease import WeightLease, after_backward
from unituner.execution.weight_source import WeightSource
from unituner.models.components.forward_override import ForwardOverride


class FrozenRegionCall[R: Tensor | tuple[Tensor, ...], **P](nn.Module):
    """Borrow a root module while invoking a particular bound computation."""

    def __init__(self, region: nn.Module, original: Callable[P, R]) -> None:
        super().__init__()
        self.region: nn.Module = region
        self.original: Callable[P, R] = original

    def invoke(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the borrowed computation while functional weights are installed."""
        return self.original(*args, **kwargs)

    forward = invoke


@dataclass(slots=True, eq=False)
class _ReplayState:
    leases: list[tuple[WeightLease, ...]] = field(default_factory=list)
    handles: list[RemovableHandle] = field(default_factory=list)
    early_hook: bool = False


class _ReplayContext:
    """Reenter checkpoint replay for retain_graph without reusing an exhausted generator context."""

    def __init__(
        self, variable: ContextVar[_ReplayState | None], state: _ReplayState, enter: Callable[[], None]
    ) -> None:
        self._variable: ContextVar[_ReplayState | None] = variable
        self._state: _ReplayState = state
        self._enter: Callable[[], None] = enter
        self._tokens: list[Token[_ReplayState | None]] = []
        self._attention_replay: AttentionReplay = AttentionReplay()

    def __enter__(self) -> None:
        token = self._variable.set(self._state)
        try:
            self._enter()
        except BaseException:
            self._variable.reset(token)
            raise
        self._tokens.append(token)
        self._attention_replay.__enter__()

    def __exit__(
        self, error_type: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self._attention_replay.__exit__(error_type, error, traceback)
        self._variable.reset(self._tokens.pop())


class _BackwardReplay[R: Tensor | tuple[Tensor, ...], **P](nn.Module):
    """Join one invocation's input gradients before releasing its recomputed VBAR leases."""

    def __init__(self, call: Callable[P, R], finish: Callable[[], None]) -> None:
        super().__init__()
        self._call: Callable[P, R] = call
        self._finish: Callable[[], None] = finish
        self.register_full_backward_hook(self._finished)

    def _finished(
        self,
        module: nn.Module,
        grad_input: Tensor | tuple[Tensor | None, ...],
        grad_output: Tensor | tuple[Tensor | None, ...],
    ) -> None:
        self._finish()

    def call(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Borrow the checkpoint call; the module backward hook owns its graph boundary."""
        return self._call(*args, **kwargs)

    forward = call


@dataclass(frozen=True, slots=True)
class _WeightSlice:
    """Locate one contiguous native parameter in an immutable host payload."""

    name: str
    parameter: nn.Parameter
    offset: int
    shape: torch.Size
    stride: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _WeightPayload:
    """Keep one dtype's source views, optional packed master and native slices."""

    host: Tensor | None
    source: WeightSource
    weights: tuple[_WeightSlice, ...]


def _pack_weights(
    parameters: tuple[tuple[str, nn.Parameter], ...], *, pinned: bool, preserve_sources: bool = False
) -> tuple[_WeightPayload, ...]:
    """Describe or pack contiguous frozen weights before mutating any parameter.

    VBAR retains CPU views instead of copying mmap-backed sources. Fixed H2D
    packs matching dtypes. Arbitrary-stride weights use the separate native
    transfer path, preserving their layout and operator selection. Copies finish
    before installation publishes CPU masters, including when sources are CUDA.
    """
    groups: dict[torch.dtype, list[tuple[str, nn.Parameter]]] = {}
    for name, parameter in parameters:
        if not parameter.requires_grad and parameter.is_contiguous():
            groups.setdefault(parameter.dtype, []).append((name, parameter))
    payloads = []
    for dtype, group in groups.items():
        host = (
            None
            if preserve_sources
            else torch.empty(
                sum(parameter.numel() for _, parameter in group), device="cpu", dtype=dtype, pin_memory=pinned
            )
        )
        chunks = []
        weights = []
        offset = 0
        for name, parameter in group:
            elements = parameter.numel()
            if host is None:
                chunks.append(parameter.detach().to("cpu").reshape(-1))
            else:
                host[offset : offset + elements].copy_(parameter.detach().reshape(-1))
            weights.append(_WeightSlice(name, parameter, offset, parameter.shape, parameter.stride()))
            offset += elements
        payloads.append(_WeightPayload(host, WeightSource(tuple(chunks) if host is None else (host,)), tuple(weights)))
    return tuple(payloads)


class FrozenBlockInstallation[R: Tensor | tuple[Tensor, ...], **P]:
    """Stage frozen CPU masters without replacing native Parameter identities.

    All contiguous weights use leases through checkpoint forward and replay.
    H2D shares a last-use cache across the model; VBAR manages physical pages.
    Both share one lookahead scheduler outside compiled native computation.
    GPU storage may be reused only after the last graph read and CUDA event.
    Noncontiguous parameters retain their original operator layout. CPU masters
    remain immutable until close; adapters and optimizer storage stay on CUDA.
    """

    def __init__(
        self,
        region: ModelRegion,
        *,
        device: torch.device,
        use_pinned_memory: bool = False,
        cache_budget_bytes: int | None = None,
        cache: H2dCache | None = None,
        use_vbar: bool = False,
        vbar_arena: VbarArena | None = None,
        prefetch: WeightPrefetch | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("Frozen block H2D residency requires CUDA")
        if cache_budget_bytes is not None and cache_budget_bytes < 0:
            raise ValueError("Frozen weight cache budget must be nonnegative")
        if use_vbar and cache_budget_bytes:
            raise ValueError("VBAR residency manages its own budget; fixed CUDA caching cannot be combined")
        if vbar_arena is not None and not use_vbar:
            raise ValueError("A VBAR arena requires use_vbar=True")
        self.region: ModelRegion = region
        self.device: torch.device = device
        original: Callable[P, R] = region.module.forward
        self._closed: bool = False
        parameters = tuple(region.module.parameters())
        if any(parameter.device not in (device, torch.device("cpu")) for parameter in parameters):
            raise ValueError("Frozen block installation requires parameters on CPU or the compute device")
        if not any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("Frozen block H2D residency requires frozen base parameters")
        self._masters: tuple[nn.Parameter, ...] = tuple(p for p in parameters if not p.requires_grad)
        self._original_devices: tuple[torch.device, ...] = tuple(p.device for p in self._masters)
        self._call: FrozenRegionCall[R, P] = FrozenRegionCall(region.module, original)
        named = tuple(self._call.named_parameters())
        self._payloads: tuple[_WeightPayload, ...] = _pack_weights(
            named, pinned=use_pinned_memory, preserve_sources=use_vbar
        )
        self._strided: tuple[tuple[str, nn.Parameter], ...] = tuple(
            (name, parameter)
            for name, parameter in named
            if not parameter.requires_grad and not parameter.is_contiguous()
        )
        self._trainable: tuple[tuple[str, nn.Parameter], ...] = tuple(
            (name, parameter) for name, parameter in named if parameter.requires_grad
        )
        # Allocate all masters before publishing storage. Keep native strides for
        # exceptional layouts rather than silently changing their GEMM dispatch.
        strided_host = tuple(parameter.detach().to("cpu", copy=not use_vbar) for _, parameter in self._strided)
        if use_pinned_memory and not use_vbar:
            strided_host = tuple(value.pin_memory() for value in strided_host)
        self._cache_owned: bool = not use_vbar and cache is None
        self._cache: H2dCache | None = (
            None
            if use_vbar
            else cache
            if cache is not None
            else H2dCache(
                device=device,
                workspace_bytes=sum(payload.source.nbytes for payload in self._payloads),
                retained_bytes=cache_budget_bytes,
            )
        )
        self._vbar_views: list[dict[str, Tensor] | None] = [None for _ in self._payloads]
        self._vbar: VbarWeights | None = (
            VbarWeights(tuple(payload.source for payload in self._payloads), device=device, arena=vbar_arena)
            if use_vbar and self._payloads
            else None
        )
        self._weights: VbarWeights | H2dWeights | None = (
            self._vbar
            if use_vbar
            else H2dWeights(tuple(payload.source for payload in self._payloads), self._cache)
            if self._cache is not None
            else None
        )
        self._prefetch: WeightPrefetch | None = prefetch
        if prefetch is not None and self._weights is not None:
            prefetch.register(self._weights)
        replay_state: ContextVar[_ReplayState | None] = ContextVar("frozen_weight_replay", default=None)
        self._replays: list[_ReplayState] = []
        for payload in self._payloads:
            for index, weight in enumerate(payload.weights):
                weight.parameter.data = (
                    payload.source.chunks[index].as_strided(
                        weight.shape, weight.stride, payload.source.chunks[index].storage_offset()
                    )
                    if payload.host is None
                    else payload.host.as_strided(weight.shape, weight.stride, weight.offset)
                )
        for (_, parameter), value in zip(self._strided, strided_host, strict=True):
            parameter.data = value

        def build(leases: tuple[WeightLease, ...]) -> dict[str, Tensor]:
            values: dict[str, Tensor] = dict(self._trainable)
            for index, payload in enumerate(self._payloads):
                transfer = leases[index].tensor
                views = self._vbar_views[index] if leases and leases[index].allocation is not None else None
                if views is None:
                    views = {
                        weight.name: transfer.as_strided(weight.shape, weight.stride, weight.offset)
                        for weight in payload.weights
                    }
                    if leases and leases[index].allocation is not None:
                        self._vbar_views[index] = views
                values.update(views)
            values.update(
                (name, parameter.to(device, non_blocking=use_pinned_memory)) for name, parameter in self._strided
            )
            values.update((name, value) for name, value in self._call.named_buffers())
            return values

        def staged(*args: P.args, **kwargs: P.kwargs) -> R:
            state = replay_state.get()
            if state is None and self._cache is not None and self._payloads:
                self._cache.prepare_forward(self._payloads[0].source)
            leases = ()
            if self._weights is not None:
                leases = self._weights.acquire() if prefetch is None else prefetch.acquire(self._weights)
            if self._weights is not None and state is not None:
                state.leases.append(leases)
            try:
                values = build(leases)
                return functional_call(self._call, values, args, kwargs, strict=True)
            finally:
                if self._weights is not None:
                    try:
                        if prefetch is not None:
                            prefetch.next(self._weights, backward=state is not None)
                    finally:
                        if state is None:
                            self._weights.release(leases)

        def installed(*args: P.args, **kwargs: P.kwargs) -> R:
            if region.module.training and torch.is_grad_enabled():
                weights = self._weights
                if weights is not None:
                    state = _ReplayState()

                    def finished() -> None:
                        for leases in state.leases:
                            weights.release(leases)
                        state.leases.clear()
                        for handle in state.handles:
                            handle.remove()
                        state.handles.clear()

                    def boundary_finished() -> None:
                        if torch.is_grad_enabled():
                            raise ValueError(
                                "Frozen block residency supports first-order backward only; create_graph=True is unsupported"
                            )
                        if not state.leases:
                            state.early_hook = True
                        finished()

                    def completed() -> None:
                        finished()
                        if state in self._replays:
                            self._replays.remove(state)

                    def enter_replay() -> None:
                        if state not in self._replays:
                            self._replays.append(state)
                        after_backward(completed)
                        weights.observe_backward()
                        if prefetch is not None:
                            after_backward(prefetch.cancel)
                        if state.early_hook and self._trainable:
                            state.handles.append(
                                torch.autograd.graph.register_multi_grad_hook(
                                    tuple(parameter for _, parameter in self._trainable), lambda gradients: finished()
                                )
                            )

                    def recorded(*args: P.args, **kwargs: P.kwargs) -> R:
                        return checkpoint(
                            staged,
                            *args,
                            use_reentrant=False,
                            preserve_rng_state=True,
                            context_fn=lambda: (nullcontext(), _ReplayContext(replay_state, state, enter_replay)),
                            **kwargs,  # ty: ignore[invalid-argument-type]
                        )

                    if any(isinstance(value, Tensor) and value.requires_grad for value in args):
                        return _BackwardReplay(recorded, boundary_finished)(*args, **kwargs)
                    # No positional input gradients: the boundary is reached at
                    # output gradients, before replay. Parameter hooks and the
                    # end-of-backward callback still own lease release.
                    result = recorded(*args, **kwargs)
                    outputs = (result,) if isinstance(result, Tensor) else result
                    differentiable = tuple(value for value in outputs if value.requires_grad)
                    if differentiable:
                        torch.autograd.graph.register_multi_grad_hook(
                            differentiable, lambda gradient: boundary_finished(), mode="any"
                        )
                    return result
            return staged(*args, **kwargs)

        self._forward = ForwardOverride(region.module, installed, owner="FrozenBlockInstallation")

    def vbar_statistics(self) -> VbarStatistics:
        """Report external physical weights without mislabeling Torch allocator reductions as VRAM savings."""
        return VbarStatistics() if self._vbar is None else self._vbar.statistics()

    def close(self) -> None:
        """Restore frozen weights to their original devices after backward, then release the owned forward."""
        if self._closed:
            return
        self._forward.check_current()
        self._vbar_views.clear()
        if self._weights is not None:
            self._weights.close()
        if self._cache_owned and self._cache is not None:
            self._cache.close()
        for state in self._replays:
            for handle in state.handles:
                handle.remove()
            state.handles.clear()
            state.leases.clear()
        self._replays.clear()
        restored = tuple(
            parameter.detach().to(device)
            for parameter, device in zip(self._masters, self._original_devices, strict=True)
        )
        for parameter, value in zip(self._masters, restored, strict=True):
            parameter.data = value
        self._forward.close()
        self._payloads = ()
        self._strided = ()
        self._trainable = ()
        self._closed = True


class FrozenModelPlacement:
    """Place CPU-loaded frozen models without materializing all blocks on CUDA at once.

    Keep frozen region parameters on CPU, move block buffers and parameters
    outside those regions to CUDA. Separate adapters are constructed directly
    on the compute device to retain native RNG initialization order. Close only
    after execution owners and backward finish; restore original placements.
    """

    def __init__(self, model: nn.Module, regions: ModelRegions, *, device: torch.device) -> None:
        if device.type != "cuda" or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Frozen model placement requires CUDA and frozen native parameters")
        region_ids = {id(parameter) for region in regions.regions for parameter in region.module.parameters()}
        values = tuple(model.parameters()) + tuple(model.buffers())
        self._originals: tuple[tuple[Tensor, torch.device], ...] = tuple((value, value.device) for value in values)
        placements = tuple(torch.device("cpu") if id(value) in region_ids else device for value in values)
        copies = tuple(value.detach().to(destination) for value, destination in zip(values, placements, strict=True))
        for value, data in zip(values, copies, strict=True):
            value.data = data
        self._closed: bool = False

    def close(self) -> None:
        """Restore only owned placement after all staged calls have completed."""
        if self._closed:
            return
        copies = tuple(value.detach().to(device) for value, device in self._originals)
        for (value, _device), data in zip(self._originals, copies, strict=True):
            value.data = data
        self._closed = True
