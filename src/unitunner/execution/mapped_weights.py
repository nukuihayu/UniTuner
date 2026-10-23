"""Borrow immutable VBAR payloads with explicit leases through checkpoint backward."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from time import monotonic
from typing import Protocol, cast

import torch
from torch import Tensor

from unitunner.execution.allocator_bootstrap import vbar_initialized
from unitunner.execution.weight_lease import PrefetchUnavailable, WeightLease, after_backward
from unitunner.execution.weight_source import WeightSource, WeightStaging


class _Signature(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, key: int) -> int: ...


class _Vbar(Protocol):
    device: int
    base_addr: int

    def alloc(self, size: int) -> tuple[_Vbar, int, int]: ...
    def fault(self, address: int, size: int) -> _Signature | None: ...
    def unpin(self, address: int, size: int) -> None: ...
    def prioritize(self) -> None: ...
    def loaded_size(self) -> int: ...
    def free_memory(self, size: int) -> int: ...
    def set_watermark(self, size: int) -> None: ...
    def get_watermark(self) -> int: ...
    def close(self) -> None: ...


class _VbarModule(Protocol):
    def ModelVBAR(self, size: int, device: int) -> _Vbar: ...


class _TorchBridge(Protocol):
    def aimdo_to_tensor(self, allocation: tuple[_Vbar, int, int], device: torch.device) -> Tensor: ...


class _Control(Protocol):
    devctxs: list[int]

    def init_devices(self, devices: list[int]) -> bool: ...


@dataclass(frozen=True, slots=True)
class VbarStatistics:
    """Report external physical residency separately from Torch allocator accounting."""

    hits: int = 0
    transfers: int = 0
    fallbacks: int = 0
    loaded_bytes: int = 0
    active_leases: int = 0
    pinned_staging_bytes: int = 0
    registered_source_bytes: int = 0


@dataclass(slots=True)
class _Payload:
    source: WeightSource
    allocation: tuple[_Vbar, int, int]
    signature: tuple[int, ...] | None = None
    device: Tensor | None = None


@dataclass(slots=True)
class _FallbackBuffer:
    """Retain immutable contents with the last-use dependency of a retired lease."""

    tensor: Tensor
    source: WeightSource | None
    ready: torch.Event | None


class VbarArena:
    """Pack one model's payloads into a single virtual range without per-block page padding.

    All visible devices get independent AIMDO contexts; allocations always use
    the explicitly resolved compute device. No user cache size, page count,
    watermark or device-capacity arithmetic is required. AIMDO manages these.
    This is per-device residency, not distributed training or pooled multi-GPU VRAM.
    """

    def __init__(
        self,
        size_bytes: int,
        *,
        device: torch.device,
        defer_release: bool = False,
        allocation_workspace_bytes: int = 0,
    ) -> None:
        if size_bytes <= 0:
            raise ValueError("VBAR arena size must be positive")
        if not vbar_initialized():
            raise ValueError("VBAR requires initialize_vbar() before importing Torch")
        if device.type != "cuda" or torch.version.hip:
            raise ValueError("The qualified VBAR training provider requires NVIDIA CUDA")
        self.device: torch.device = torch.device(
            "cuda", torch.cuda.current_device() if device.index is None else device.index
        )
        control = cast(_Control, import_module("comfy_aimdo.control"))
        if not control.devctxs and not control.init_devices(list(range(torch.cuda.device_count()))):
            raise RuntimeError("AIMDO device contexts could not initialize")
        self._bridge: _TorchBridge = cast(_TorchBridge, import_module("comfy_aimdo.torch"))
        api = cast(_VbarModule, import_module("comfy_aimdo.model_vbar"))
        # CUDA's current context must match the requested device before faulting.
        with torch.cuda.device(self.device):
            # Torch can select a device without creating its primary context.
            # AIMDO's first budget poll otherwise fails with invalid context
            # and may lower the watermark to zero until its next poll.
            torch.cuda.mem_get_info(self.device)
            self._owner: _Vbar = api.ModelVBAR(size_bytes, self.device.index)
            self._owner.prioritize()
        self.staging: WeightStaging = WeightStaging(device=self.device)
        self._closed: bool = False
        self.defer_release: bool = defer_release
        self._retired: deque[tuple[torch.Event, tuple[WeightLease, ...]]] = deque()
        self._buffers: list[_FallbackBuffer] = []
        self._allocation_workspace_bytes: int = allocation_workspace_bytes
        self._workspace_bytes: int = 0
        self._allocation_retries: int = torch.cuda.memory_stats(self.device)["num_alloc_retries"]
        self._backward_pending: bool = False
        self._planned: deque[tuple[tuple[_Vbar, int, int], ...]] | None = None
        self._free_at_miss: int | None = None
        self._recovery_check_seconds: float = 0.0

    def observe_miss(self) -> None:
        """Record pressure once for the shared arena, rather than once per block."""
        if self._free_at_miss is None:
            self._free_at_miss = torch.cuda.mem_get_info(self.device)[0]
            self._recovery_check_seconds = monotonic() + 1.0

    def recover(self) -> None:
        """Reconsider residency after free memory grows by a page-rounded region.

        All regions share one watermark, so independent per-block polls would
        repeat driver queries and could reset it repeatedly during one traversal.
        """
        if self._free_at_miss is None:
            return
        now = monotonic()
        if now < self._recovery_check_seconds:
            return
        self._recovery_check_seconds = now + 1.0
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        page_bytes = 32 * 1024**2
        required = (self._workspace_bytes + page_bytes - 1) // page_bytes * page_bytes
        if free_bytes >= self._free_at_miss + required:
            self._owner.prioritize()
            self._free_at_miss = None

    def plan_regions(self, sizes: Sequence[Sequence[int]]) -> None:
        """Spread address-prefix residency across logical execution instead of streaming a suffix.

        Bit-reversed priority distributes retained prefixes across the model.
        Execution and payload arithmetic stay unchanged. Call once before region
        construction, with sizes in region/dtype order.
        """
        if self._planned is not None:
            raise RuntimeError("VBAR region addresses have already been planned")
        bits = max(1, (len(sizes) - 1).bit_length())
        order = sorted(range(len(sizes)), key=lambda index: int(f"{index:0{bits}b}"[::-1], 2))
        allocations: dict[int, tuple[tuple[_Vbar, int, int], ...]] = {}
        for index in order:
            allocations[index] = tuple(self._owner.alloc(size) for size in sizes[index])
        self._planned = deque(allocations[index] for index in range(len(sizes)))

    def allocate(self, sizes: tuple[int, ...]) -> tuple[tuple[_Vbar, int, int], ...]:
        """Consume one region's addresses, or allocate an unplanned standalone owner."""
        self._workspace_bytes = max(self._workspace_bytes, sum(sizes))
        if self._planned is None:
            return tuple(self._owner.alloc(size) for size in sizes)
        if not self._planned:
            raise ValueError("VBAR region allocation exceeds the planned payload inventory")
        allocations = self._planned.popleft()
        if tuple(value[2] for value in allocations) != sizes:
            raise ValueError("VBAR region payload sizes differ from the planned inventory")
        return allocations

    def observe_backward(self) -> None:
        """Back off fresh speculative allocations after real allocator retries."""
        if not self._backward_pending:
            self._backward_pending = True
            after_backward(self._completed_backward)

    def _completed_backward(self) -> None:
        self._backward_pending = False
        retries = torch.cuda.memory_stats(self.device)["num_alloc_retries"]
        if retries > self._allocation_retries:
            self._allocation_workspace_bytes = max(self._allocation_workspace_bytes, self._workspace_bytes)
        self._allocation_retries = retries

    def borrow(self, source: WeightSource, *, speculative: bool = False) -> WeightLease:
        """Reuse retired cast storage while keeping every live graph's weights immutable.

        Two idle buffers cover current/lookahead execution. Outstanding backward
        leases may require additional storage, which must never be overwritten.
        A retired buffer carries its compute event into the next copy stream;
        reuse adds a device dependency without blocking the submitting CPU.
        An exact immutable source match also preserves its contents, avoiding
        another H2D at execution direction changes. Other borrows use best fit.
        """
        self.drain()
        index = next((index for index, value in enumerate(self._buffers) if value.source is source), None)
        if index is None:
            index = min(
                (index for index, value in enumerate(self._buffers) if value.tensor.nbytes >= source.nbytes),
                key=lambda index: self._buffers[index].tensor.nbytes,
                default=None,
            )
        populated = False
        if index is None:
            if speculative:
                free, _ = torch.cuda.mem_get_info(self.device)
                available = free + torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
                if source.nbytes + self._allocation_workspace_bytes > available:
                    raise PrefetchUnavailable
            buffer = torch.empty(source.nbytes, dtype=torch.uint8, device=self.device)
        else:
            retained = self._buffers.pop(index)
            buffer = retained.tensor
            populated = retained.source is source
            if retained.ready is not None:
                torch.cuda.current_stream(self.device).wait_event(retained.ready)
            buffer.record_stream(torch.cuda.current_stream(self.device))

        def recycle(ready: torch.Event | None, populated: bool) -> None:
            if len(self._buffers) < 2:
                self._buffers.append(_FallbackBuffer(buffer, source if populated else None, ready))

        return WeightLease(buffer[: source.nbytes].view(source.dtype), None, recycle=recycle, populated=populated)

    def drain(self) -> None:
        """Unpin retired payloads only after their last compute-stream read completes."""
        while self._retired and self._retired[0][0].query():
            _, leases = self._retired.popleft()
            for lease in leases:
                lease.release()

    def retire(self, leases: tuple[WeightLease, ...]) -> None:
        """Keep in-flight immutable pages pinned across asynchronous stream execution."""
        if not leases:
            return
        ready = torch.cuda.current_stream(self.device).record_event()
        mapped = tuple(lease for lease in leases if lease.allocation is not None)
        for lease in leases:
            if lease.allocation is None:
                lease.release(ready=ready)
        if mapped and self.defer_release:
            self._retired.append((ready, mapped))
            if len(self._retired) > 1:
                self._retired[0][0].synchronize()
                self.drain()
        else:
            for lease in mapped:
                lease.release()

    def wait_retired(self) -> None:
        """Recycle previous compute storage before loading another lookahead payload."""
        if self._retired:
            self._retired[-1][0].synchronize()
            self.drain()

    def loaded_bytes(self) -> int:
        """Report physical bytes once per arena, including shared 32MiB pages."""
        return 0 if self._closed else self._owner.loaded_size()

    def close(self) -> None:
        """Free the range after all block owners and their backward leases have closed."""
        if self._closed:
            return
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            self.drain()
            self.staging.close()
            self._buffers.clear()
            self._owner.close()
        self._closed = True


class VbarWeights:
    """Borrow a region's immutable payloads from an owned or model-shared VBAR arena."""

    prefetch_past_resident: bool = True

    def __init__(
        self, hosts: Sequence[Tensor | WeightSource], *, device: torch.device, arena: VbarArena | None = None
    ) -> None:
        if not hosts:
            raise ValueError("VBAR requires nonempty CPU weight sources")
        sources = tuple(WeightSource((host,)) if isinstance(host, Tensor) else host for host in hosts)
        self._owns_arena: bool = arena is None
        self._arena: VbarArena = (
            VbarArena(sum((host.nbytes + 511) // 512 * 512 for host in sources), device=device)
            if arena is None
            else arena
        )
        self.device: torch.device = self._arena.device
        if device.type != "cuda" or (device.index is not None and device.index != self.device.index):
            raise ValueError("VBAR payload and arena devices must match")
        self._owner: _Vbar = self._arena._owner
        self._bridge: _TorchBridge = self._arena._bridge
        self._payloads: tuple[_Payload, ...] = tuple(
            _Payload(source, allocation)
            for source, allocation in zip(
                sources, self._arena.allocate(tuple(source.nbytes for source in sources)), strict=True
            )
        )
        self._active: set[WeightLease] = set()
        self.hits: int = 0
        self.transfers: int = 0
        self.fallbacks: int = 0
        self._closed: bool = False

    def observe_backward(self) -> None:
        """Share one allocator observation across all regions in this arena."""
        self._arena.observe_backward()

    def needs_transfer(self) -> bool:
        """Predict misses without pinning future pages; demand still validates signatures.

        AIMDO evicts address suffixes. This hint lets lookahead skip resident
        computation and start a streamed block earlier. It is never used as
        proof that a device address is safe to read.
        """
        watermark = self._owner.get_watermark() * 32 * 1024**2
        return any(
            payload.signature is None
            or payload.allocation[1] - self._owner.base_addr + payload.allocation[2] > watermark
            for payload in self._payloads
        )

    def acquire(self, *, speculative: bool = False) -> tuple[WeightLease, ...]:
        """Fault and populate changed signatures only; never touch a failed fault's virtual address."""
        if self._closed:
            raise RuntimeError("VBAR weights are closed")
        leases = []
        try:
            with torch.cuda.device(self.device):
                self._arena.drain()
                self._arena.recover()
                for payload in self._payloads:
                    owner, address, size = payload.allocation
                    signature = owner.fault(address, size)
                    if signature is None:
                        self._arena.observe_miss()
                        lease = self._arena.borrow(payload.source, speculative=speculative)
                        leases.append(lease)
                        self._active.add(lease)
                        if not lease.populated:
                            self._arena.staging.copy(payload.source, lease.tensor, streaming=True)
                            lease.populated = True
                            self.transfers += 1
                        else:
                            self.hits += 1
                        self.fallbacks += 1
                    else:
                        if payload.device is None:
                            try:
                                payload.device = self._bridge.aimdo_to_tensor(payload.allocation, self.device).view(
                                    payload.source.dtype
                                )
                            except BaseException:
                                # Fault already pinned these pages, but no lease
                                # owns them until the tensor view exists.
                                owner.unpin(address, size)
                                raise
                        value = payload.device
                        lease = WeightLease(value, payload.allocation)
                        leases.append(lease)
                        self._active.add(lease)
                        serial = tuple(signature[index] for index in range(len(signature)))
                        if serial != payload.signature:
                            self._arena.staging.copy(payload.source, value)
                            payload.signature = serial
                            self.transfers += 1
                        else:
                            self.hits += 1
                        continue
        except BaseException:
            self.release(tuple(leases))
            raise
        return tuple(leases)

    def statistics(self) -> VbarStatistics:
        """Return aggregate hit/copy counters and actual AIMDO-resident physical bytes."""
        loaded = 0 if self._closed or not self._owns_arena else self._arena.loaded_bytes()
        return VbarStatistics(
            self.hits,
            self.transfers,
            self.fallbacks,
            loaded,
            len(self._active),
            self._arena.staging.pinned_bytes if self._owns_arena else 0,
            self._arena.staging.registered_bytes if self._owns_arena else 0,
        )

    def release(self, leases: tuple[WeightLease, ...]) -> None:
        """Release a completed invocation's pins and remove active ownership."""
        with torch.cuda.device(self.device):
            retired = tuple(lease for lease in leases if lease in self._active)
            for lease in retired:
                self._active.discard(lease)
            self._arena.retire(retired)

    def close(self) -> None:
        """Drain exceptional leases, free addresses explicitly, and prevent upstream finalizer double-free."""
        if self._closed:
            return
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            self.release(tuple(self._active))
            if self._owns_arena:
                self._arena.close()
        self._payloads = ()
        self._closed = True
