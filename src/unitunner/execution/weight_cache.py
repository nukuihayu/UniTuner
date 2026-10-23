"""Retain distributed frozen regions and stream misses without overwriting graph reads."""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from unitunner.execution.weight_lease import PrefetchUnavailable, WeightLease, after_backward
from unitunner.execution.weight_source import WeightSource


@dataclass(slots=True, eq=False)
class _Buffer:
    tensor: Tensor
    source: WeightSource | None = None
    users: int = 0
    loaded: torch.Event | None = None
    readers: list[torch.Event] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class H2dStatistics:
    """Separate retained storage, active graph storage and actual weight copies."""

    hits: int
    transfers: int
    allocated_bytes: int
    active_bytes: int
    capacity_bytes: int
    automatic: bool


class H2dCache:
    """Share a bounded cache between blocks, forward and backward.

    The budget includes the current block's workspace in addition to an optional
    explicit retained-weight budget. Idle buffers are reused only after their last
    CUDA readers; live leases are never overwritten. Demand may exceed the cache
    budget for outstanding graphs, whereas speculative prefetch must fit it.
    Automatic capacity starts with current-block storage and is adjusted at
    backward completion using observed allocator peaks and device availability.
    Admission protects distributed resident regions; remaining buffers use LRU.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        workspace_bytes: int,
        retained_bytes: int | None = None,
        cast_workspace_bytes: int = 0,
        allocation_workspace_bytes: int = 0,
    ) -> None:
        self.device: torch.device = device
        self._workspace: int = workspace_bytes
        self._allocation_workspace: int = allocation_workspace_bytes
        self._reserve: int = workspace_bytes + cast_workspace_bytes
        self._first_source: WeightSource | None = None
        self._automatic: bool = retained_bytes is None
        self._capacity: int = workspace_bytes if retained_bytes is None else workspace_bytes + retained_bytes
        self._buffers: list[_Buffer] = []
        self._sources: dict[int, _Buffer] = {}
        self._regions: list[tuple[WeightSource, ...]] = []
        self._resident: set[int] = set()
        self._backward_pending: bool = False
        self._closed: bool = False
        self.hits: int = 0
        self.transfers: int = 0

    def register(self, sources: tuple[WeightSource, ...]) -> None:
        """Plan whole-region admission before training, without allocating CUDA storage."""
        if sources:
            self._regions.append(sources)
            self._plan_residency()

    def _plan_residency(self) -> None:
        """Distribute retained regions while leaving two workspaces for streaming.

        Cached resident neighbors can then be crossed safely by lookahead:
        loading a distant miss cannot evict a region the compute stream expects
        to find resident. Active graph leases remain protected independently.
        """
        sizes = [sum(source.nbytes for source in region) for region in self._regions]
        selected: list[int] = []
        if sum(sizes) <= self._capacity:
            selected = list(range(len(sizes)))
        else:
            budget = max(0, self._capacity - 2 * self._workspace)
            for count in range(len(sizes) - 1, 0, -1):
                indices = [((2 * index + 1) * len(sizes)) // (2 * count) for index in range(count)]
                if sum(sizes[index] for index in indices) <= budget:
                    selected = indices
                    break
        self._resident = {id(source) for index in selected for source in self._regions[index]}

    def retained(self, sources: tuple[WeightSource, ...]) -> bool:
        """Report whether lookahead may cross this region without evicting it."""
        return all(id(source) in self._resident for source in sources)

    def contains(self, source: WeightSource) -> bool:
        """Check source identity without changing last-use order."""
        return id(source) in self._sources

    def _discard(self, buffer: _Buffer) -> None:
        if buffer.source is not None:
            self._sources.pop(id(buffer.source), None)
        self._buffers.remove(buffer)

    def _trim(self, *, preserve: WeightSource | None = None) -> None:
        """Shrink idle storage in LRU order, evicting admitted residents last."""
        retained = sum(value.tensor.nbytes for value in self._buffers)
        if retained <= self._capacity:
            return
        for buffer in sorted(self._buffers, key=lambda value: id(value.source) in self._resident):
            if not buffer.users and (preserve is None or buffer.source is not preserve):
                retained -= buffer.tensor.nbytes
                self._discard(buffer)
                if retained <= self._capacity:
                    break

    def prepare_forward(self, source: WeightSource) -> None:
        """Shrink once before a forward; temporary block allocations are not spare cache."""
        if self._first_source is None:
            self._first_source = source
        if source is not self._first_source:
            return
        if self._automatic:
            # Shrink before a larger bucket or another process consumes memory;
            # growth is allowed only after a completed backward calibration.
            free, _ = torch.cuda.mem_get_info(self.device)
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            retained = sum(value.tensor.nbytes for value in self._buffers)
            self._capacity = min(
                self._capacity, max(self._workspace, retained + free + reserved - allocated - self._reserve)
            )
            self._plan_residency()
            self._trim(preserve=source)

    def acquire(self, source: WeightSource, *, speculative: bool) -> WeightLease:
        """Borrow immutable contents or overwrite an idle, event-protected buffer."""
        if self._closed:
            raise RuntimeError("Frozen weight cache is closed")
        stream = torch.cuda.current_stream(self.device)
        buffer = self._sources.get(id(source))
        if buffer is not None:
            self.hits += 1
            self._buffers.remove(buffer)
            self._buffers.append(buffer)
        else:
            # Reuse least-recently-used storage; prefer the smallest fitting idle
            # allocation for mixed dtypes and unequal region sizes.
            idle = [value for value in self._buffers if not value.users and id(value.source) not in self._resident]
            allocated = sum(value.tensor.nbytes for value in self._buffers)
            buffer = (
                None
                if allocated + source.nbytes <= self._capacity
                else min(
                    (value for value in idle if value.tensor.nbytes >= source.nbytes),
                    key=lambda value: value.tensor.nbytes,
                    default=None,
                )
            )
            if buffer is None:
                allocated = sum(value.tensor.nbytes for value in self._buffers)
                while idle and allocated + source.nbytes > self._capacity:
                    old = idle.pop(0)
                    allocated -= old.tensor.nbytes
                    self._discard(old)
                if speculative and allocated + source.nbytes > self._capacity:
                    raise PrefetchUnavailable
                if speculative:
                    free, _ = torch.cuda.mem_get_info(self.device)
                    available = (
                        free + torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
                    )
                    if source.nbytes > available:
                        raise PrefetchUnavailable
                buffer = _Buffer(torch.empty(source.nbytes, device=self.device, dtype=torch.uint8))
                self._buffers.append(buffer)
            else:
                if buffer.source is not None:
                    self._sources.pop(id(buffer.source), None)
                self._buffers.remove(buffer)
                self._buffers.append(buffer)
                for ready in buffer.readers:
                    stream.wait_event(ready)
                buffer.readers.clear()
            buffer.source = None
            buffer.tensor.record_stream(stream)
            try:
                destination = buffer.tensor[: source.nbytes].view(source.dtype)
                offset = 0
                for chunk in source.chunks:
                    destination[offset : offset + chunk.numel()].copy_(chunk, non_blocking=chunk.is_pinned())
                    offset += chunk.numel()
            finally:
                # Even a partially submitted failed copy must finish before reuse.
                buffer.loaded = stream.record_event()
                buffer.readers.append(buffer.loaded)
            buffer.source = source
            self._sources[id(source)] = buffer
            self.transfers += 1
        if buffer.loaded is not None:
            stream.wait_event(buffer.loaded)
        buffer.tensor.record_stream(stream)
        buffer.users += 1

        def recycle(ready: torch.Event | None, populated: bool) -> None:
            buffer.users -= 1
            buffer.readers[:] = [event for event in buffer.readers if not event.query()]
            buffer.readers.append(ready if ready is not None else torch.cuda.current_stream(self.device).record_event())
            self._trim()

        return WeightLease(buffer.tensor[: source.nbytes].view(source.dtype), None, recycle=recycle, populated=True)

    def observe_backward(self) -> None:
        """Sample once per engine backward, without resetting global allocator peaks."""
        if self._automatic and not self._backward_pending:
            self._backward_pending = True
            after_backward(self._completed_backward)

    def _completed_backward(self) -> None:
        self._backward_pending = False
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        free, _ = torch.cuda.mem_get_info(self.device)
        retained = sum(value.tensor.nbytes for value in self._buffers)
        # Keep at least one whole region for unobserved replay/cast work. The
        # observed transient peak can demand more; unrelated global peaks only
        # make this conservative. Leave one largest tensor allocation of slack:
        # allocator free fragments are not necessarily one contiguous workspace.
        self._reserve = max(self._reserve, peak - allocated + self._allocation_workspace)
        self._capacity = max(self._workspace, retained + free + reserved - allocated - self._reserve)
        self._plan_residency()
        self._trim()

    def statistics(self) -> H2dStatistics:
        """Count every live buffer once even when several leases share a source."""
        return H2dStatistics(
            self.hits,
            self.transfers,
            sum(value.tensor.nbytes for value in self._buffers),
            sum(value.tensor.nbytes for value in self._buffers if value.users),
            self._capacity,
            self._automatic,
        )

    def close(self) -> None:
        """Finish all CUDA users before dropping the shared weight cache."""
        if not self._closed:
            torch.cuda.synchronize(self.device)
            self._sources.clear()
            self._buffers.clear()
            self._regions.clear()
            self._resident.clear()
            self._closed = True


class H2dWeights:
    """Bind one region's immutable CPU payloads to a shared residency cache."""

    def __init__(self, sources: tuple[WeightSource, ...], cache: H2dCache) -> None:
        self.device: torch.device = cache.device
        self._sources: tuple[WeightSource, ...] = sources
        self._cache: H2dCache = cache
        cache.register(sources)
        self._active: set[WeightLease] = set()

    @property
    def prefetch_past_resident(self) -> bool:
        """Cross only admitted residents; ordinary cached neighbors remain evictable."""
        return self._cache.retained(self._sources)

    def needs_transfer(self) -> bool:
        """Predict missing payloads without acquiring speculative graph lifetimes."""
        return any(not self._cache.contains(source) for source in self._sources)

    def observe_backward(self) -> None:
        """Let the shared cache calibrate once after the complete graph finishes."""
        self._cache.observe_backward()

    def acquire(self, *, speculative: bool = False) -> tuple[WeightLease, ...]:
        """Release partial acquisitions on capacity or transfer failure."""
        leases: list[WeightLease] = []
        try:
            for source in self._sources:
                lease = self._cache.acquire(source, speculative=speculative)
                leases.append(lease)
                self._active.add(lease)
        except BaseException:
            self.release(tuple(leases))
            raise
        return tuple(leases)

    def release(self, leases: tuple[WeightLease, ...]) -> None:
        """Record the last compute-stream read before recycling any GPU storage."""
        if leases:
            ready = torch.cuda.current_stream(self.device).record_event()
            for lease in leases:
                if lease in self._active:
                    self._active.remove(lease)
                    lease.release(ready=ready)

    def close(self) -> None:
        """Retire exceptional active leases before the shared cache closes."""
        self.release(tuple(self._active))
