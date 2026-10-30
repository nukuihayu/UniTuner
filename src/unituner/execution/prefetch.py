"""Prefetch frozen weight misses within available fallback capacity."""

import torch

from unituner.execution.mapped_weights import VbarArena
from unituner.execution.weight_lease import FrozenWeights, PrefetchUnavailable, WeightLease


class WeightPrefetch:
    """Own bounded lookahead leases; keep active backward storage immutable.

    Registered region order is a prediction, never a requirement. Unexpected
    branches discard the unused prediction and acquire the actual block normally.
    Speculative allocation failure defers loading until demand. No user lookahead
    or ring size is required. Providers own capacity decisions.
    """

    def __init__(self, device: torch.device, *, arena: VbarArena | None = None) -> None:
        self._arena: VbarArena | None = arena
        self._stream: torch.cuda.Stream = torch.cuda.Stream(device=device)
        self._weights: list[FrozenWeights] = []
        self._indices: dict[FrozenWeights, int] = {}
        self._pending: list[tuple[FrozenWeights, tuple[WeightLease, ...], torch.Event]] = []
        self._closed: bool = False

    def register(self, weights: FrozenWeights) -> None:
        """Append an owner while constructing the model's ordered execution regions."""
        if self._closed or weights in self._indices:
            raise RuntimeError("Cannot register a closed schedule or duplicate weight owner")
        self._indices[weights] = len(self._weights)
        self._weights.append(weights)

    def acquire(self, weights: FrozenWeights) -> tuple[WeightLease, ...]:
        """Consume a matching prediction while retaining future loads across resident work."""
        prepared = next((entry for entry in self._pending if entry[0] is weights), None)
        if prepared is None:
            if self._pending and weights.needs_transfer():
                self.cancel()
            if not self._pending and self._arena is not None:
                self._arena.wait_retired()
            return weights.acquire()
        self._pending.remove(prepared)
        _, leases, ready = prepared
        stream = torch.cuda.current_stream(weights.device)
        stream.wait_event(ready)
        for lease in leases:
            if lease.allocation is None:
                lease.tensor.record_stream(stream)
        return leases

    def next(self, weights: FrozenWeights, *, backward: bool) -> None:
        """Prefetch the next missing block without extending the speculative queue."""
        direction = -1 if backward else 1
        current = self._indices[weights]
        if any((self._indices[owner] - current) * direction <= 0 for owner, _, _ in self._pending):
            self.cancel()
        index = current + direction
        while not self._pending and 0 <= index < len(self._weights):
            target = self._weights[index]
            index += direction
            if not target.needs_transfer():
                if not target.prefetch_past_resident:
                    return
                continue
            if self._arena is not None:
                self._arena.drain()
            try:
                with torch.cuda.stream(self._stream):
                    leases = target.acquire(speculative=True)
                    ready = self._stream.record_event()
            except (PrefetchUnavailable, torch.OutOfMemoryError) as error:
                # Demand acquisition reports real OOM; speculative loading
                # may be cancelled when available memory is insufficient.
                from unituner.logging import get_logger

                get_logger("execution.prefetch").debug("Speculative prefetch cancelled: %s", error)
                return
            self._pending.append((target, leases, ready))

    def cancel(self) -> None:
        """Finish unused transfers before releasing their immutable leases."""
        pending, self._pending = self._pending, []
        for owner, leases, ready in pending:
            ready.synchronize()
            owner.release(leases)

    def close(self) -> None:
        """Finish copies before any registered weight owner is closed."""
        if not self._closed:
            self.cancel()
            self._stream.synchronize()
            self._weights.clear()
            self._indices.clear()
            self._closed = True
