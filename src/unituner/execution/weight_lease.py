"""Share graph lifetimes and scheduling contracts across frozen-weight providers."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor


class _Engine(Protocol):
    def queue_callback(self, callback: Callable[[], None]) -> None: ...


def after_backward(callback: Callable[[], None]) -> None:
    """Queue fallback lease cleanup at graph completion through pinned Torch2.13's engine.

    Public module backward hooks release leases earlier. The engine callback is
    necessary for regions without differentiable positional inputs. No public
    graph-completion callback API exists in this supported Torch version.
    """
    if not torch.__version__.startswith("2.13."):
        raise RuntimeError("Frozen weight backward lifetime requires the qualified Torch2.13 engine callback")
    engine = cast(_Engine, torch.autograd.Variable._execution_engine)
    engine.queue_callback(callback)


@dataclass(slots=True, eq=False)
class WeightLease:
    """Keep immutable storage alive until backward finishes reading it.

    Recycling carries the last-use event to the H2D cache.
    """

    tensor: Tensor
    released: bool = False
    recycle: Callable[[torch.Event | None], None] | None = None

    def release(self, *, ready: torch.Event | None = None) -> None:
        """Recycle once after the caller finishes all graph reads."""
        if not self.released:
            if self.recycle is not None:
                self.recycle(ready)
            self.released = True


class PrefetchUnavailable(Exception):
    """Speculative storage would exceed the selected cache budget."""


class FrozenWeights(Protocol):
    """Expose H2D weight storage to the lookahead scheduler."""

    device: torch.device

    @property
    def prefetch_past_resident(self) -> bool: ...

    def acquire(self, *, speculative: bool = False) -> tuple[WeightLease, ...]: ...
    def release(self, leases: tuple[WeightLease, ...]) -> None: ...
    def needs_transfer(self) -> bool: ...
    def observe_backward(self) -> None: ...
