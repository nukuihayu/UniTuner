"""Preserve immutable CPU weight sources and bound pinned transfer staging."""

import sys
from dataclasses import dataclass
from mmap import PAGESIZE
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class WeightSource:
    """Describe a packed device payload without copying its CPU or mmap sources.

    Chunks are contiguous, flat, immutable CPU views of the same dtype. Their
    concatenation defines the device layout; views retain the source storage.
    """

    chunks: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if not self.chunks or any(
            value.device.type != "cpu"
            or value.ndim != 1
            or not value.is_contiguous()
            or value.dtype != self.chunks[0].dtype
            for value in self.chunks
        ):
            raise ValueError("Weight sources require matching contiguous flat CPU tensors")

    @property
    def dtype(self) -> torch.dtype:
        """Return the common source and destination dtype."""
        return self.chunks[0].dtype

    @property
    def nbytes(self) -> int:
        """Return packed payload size in bytes."""
        return sum(value.nbytes for value in self.chunks)


@dataclass(slots=True)
class _Slot:
    host: Tensor
    ready: torch.Event | None = None


@dataclass(frozen=True, slots=True)
class _Registration:
    address: int
    size: int
    sources: tuple[Tensor, ...]


def _registration_budget() -> int:
    """Leave OS headroom and cap Linux source registration at one quarter of RAM."""
    if sys.platform != "linux":
        return 0
    memory = {
        key: int(value.split()[0]) * 1024
        for key, value in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    }
    total, available = memory["MemTotal"], memory["MemAvailable"]
    limit_path = Path("/sys/fs/cgroup/memory.max")
    current_path = Path("/sys/fs/cgroup/memory.current")
    if limit_path.is_file() and current_path.is_file():
        limit = limit_path.read_text().strip()
        if limit != "max":
            total = min(total, int(limit))
            available = min(available, int(limit) - int(current_path.read_text()))
    return max(0, min(total // 4, available - 2 * 1024**3))


class WeightStaging:
    """Own bounded source registrations and two lazy 64MiB pinned windows.

    Only misses touch CPU sources. Repeated fallbacks register original source
    pages on Linux within an automatic RAM budget, avoiding repeated packing.
    A window is reusable once its H2D event
    completes, independently of the destination's backward lifetime. Device
    payloads remain owned by their leases; this ring never overwrites them.
    Close after all transfers, before releasing the CPU source owners.
    """

    def __init__(self, *, device: torch.device) -> None:
        self.device: torch.device = device
        self._slots: list[_Slot] = []
        self._next: int = 0
        self._window_bytes: int = 64 * 1024**2
        self._registrations: list[_Registration] = []
        self._registration_budget: int | None = None
        self._attempted: set[WeightSource] = set()

    def _pinned(self, chunk: Tensor) -> bool:
        """Distinguish full coverage from a tensor beginning on a shared pinned page."""
        start, end = chunk.data_ptr(), chunk.data_ptr() + chunk.nbytes
        if not any(
            value.address // PAGESIZE * PAGESIZE
            <= start
            < (value.address + value.size + PAGESIZE - 1) // PAGESIZE * PAGESIZE
            for value in self._registrations
        ):
            return chunk.is_pinned()
        for value in sorted(self._registrations, key=lambda value: value.address):
            if value.address > start:
                break
            start = max(start, value.address + value.size)
            if start >= end:
                return True
        return False

    @property
    def registered_bytes(self) -> int:
        """Return original source pages registered on demand, without a CPU copy."""
        return sum(value.size for value in self._registrations)

    def _register(self, source: WeightSource) -> None:
        """Pin streaming sources in place within an automatic budget, retaining owners.

        Merge shared page intervals: adjacent parameters may share host pages,
        and CUDA rejects overlapping registrations. Cold VBAR loads
        do not register sources; this cache is reserved for repeated fallbacks.
        """
        if self._registration_budget is None:
            self._registration_budget = _registration_budget()
        if source in self._attempted:
            return
        self._attempted.add(source)
        ranges: list[tuple[int, int]] = []
        for chunk in source.chunks:
            if self._pinned(chunk) or not chunk.numel():
                continue
            # CUDA accepts byte ranges. Rounding into neighbouring malloc
            # allocations makes unrelated tensors partly registered, which
            # can invalidate their ordinary CUDA copies.
            start = chunk.data_ptr()
            end = start + chunk.nbytes
            ranges.append((start, end))
        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        intervals: list[tuple[int, int, tuple[_Registration, ...]]] = [(start, end, ()) for start, end in merged]
        intervals.extend((value.address, value.address + value.size, (value,)) for value in self._registrations)
        components: list[tuple[int, int, tuple[_Registration, ...]]] = []
        for start, end, touched in sorted(intervals, key=lambda value: value[0]):
            if components and start <= components[-1][1]:
                left, right, prior = components[-1]
                components[-1] = (left, max(right, end), prior + touched)
            else:
                components.append((start, end, touched))
        plans = [
            (start, end, touched)
            for start, end, touched in components
            if not (len(touched) == 1 and touched[0].address == start and touched[0].size == end - start)
        ]
        required = sum(end - start - sum(value.size for value in touched) for start, end, touched in plans)
        if required > self._registration_budget - self.registered_bytes:
            return
        for start, end, touched in plans:
            if touched:
                # Re-register the union rather than splitting a native tensor
                # between independent registrations: even its ordinary .to()
                # must remain valid outside our transfer helper.
                torch.cuda.synchronize(self.device)
            owners = list(source.chunks)
            for registered in touched:
                runtime = torch.cuda.cudart()
                assert runtime is not None
                result = runtime.cudaHostUnregister(registered.address)
                if result != 0:
                    raise RuntimeError(f"CUDA source unregistration failed with status {result}")
                owners.extend(registered.sources)
                self._registrations.remove(registered)
            runtime = torch.cuda.cudart()
            assert runtime is not None
            result = runtime.cudaHostRegister(start, end - start, 1)
            if result != 0:
                raise RuntimeError(f"CUDA source registration failed with status {result}")
            self._registrations.append(_Registration(start, end - start, tuple(owners)))

    @property
    def pinned_bytes(self) -> int:
        """Return live staging bytes, excluding caller-owned pinned sources."""
        return sum(slot.host.nbytes for slot in self._slots)

    def copy(self, source: WeightSource, destination: Tensor, *, streaming: bool = False) -> None:
        """Gather CPU views into an immutable CUDA destination with bounded pinning."""
        if (
            destination.device != self.device
            or destination.nbytes != source.nbytes
            or destination.dtype != source.dtype
        ):
            raise ValueError("Weight staging destination must match source size, dtype and compute device")
        if streaming:
            self._register(source)
        if all(self._pinned(chunk) for chunk in source.chunks):
            offset = 0
            target = destination.view(torch.uint8)
            for chunk in source.chunks:
                raw = chunk.view(torch.uint8)
                start = 0
                while start < chunk.nbytes:
                    address = chunk.data_ptr() + start
                    registration = next(
                        (
                            value
                            for value in self._registrations
                            if value.address <= address < value.address + value.size
                        ),
                        None,
                    )
                    # CUDA memcpy cannot span independent host registrations,
                    # even when the covered page intervals are contiguous.
                    length = (
                        chunk.nbytes - start
                        if registration is None
                        else min(chunk.nbytes - start, registration.address + registration.size - address)
                    )
                    target[offset + start : offset + start + length].copy_(
                        raw[start : start + length], non_blocking=True
                    )
                    start += length
                offset += chunk.nbytes
            return
        chunks = tuple(chunk.view(torch.uint8) for chunk in source.chunks)
        target = destination.view(torch.uint8)
        chunk_index = 0
        chunk_offset = 0
        for offset in range(0, source.nbytes, self._window_bytes):
            count = min(self._window_bytes, source.nbytes - offset)
            if len(self._slots) < 2:
                slot = _Slot(torch.empty(min(source.nbytes, self._window_bytes), dtype=torch.uint8, pin_memory=True))
                self._slots.append(slot)
            else:
                slot = self._slots[self._next]
            self._next = (self._next + 1) % 2
            if slot.ready is not None:
                slot.ready.synchronize()
            if slot.host.nbytes < count:
                slot.host = torch.empty(min(source.nbytes, self._window_bytes), dtype=torch.uint8, pin_memory=True)
            written = 0
            while written < count:
                chunk = chunks[chunk_index]
                length = min(count - written, chunk.numel() - chunk_offset)
                slot.host[written : written + length].copy_(chunk[chunk_offset : chunk_offset + length])
                written += length
                chunk_offset += length
                if chunk_offset == chunk.numel():
                    chunk_index += 1
                    chunk_offset = 0
            target[offset : offset + count].copy_(slot.host[:count], non_blocking=True)
            slot.ready = torch.cuda.current_stream(self.device).record_event()

    def close(self) -> None:
        """Drain H2D reads before dropping the bounded pinned windows."""
        for slot in self._slots:
            if slot.ready is not None:
                slot.ready.synchronize()
        if self._registrations:
            torch.cuda.synchronize(self.device)
        while self._registrations:
            registered = self._registrations[-1]
            runtime = torch.cuda.cudart()
            assert runtime is not None
            result = runtime.cudaHostUnregister(registered.address)
            if result != 0:
                raise RuntimeError(f"CUDA source unregistration failed with status {result}")
            self._registrations.pop()
        self._attempted.clear()
        self._slots.clear()
