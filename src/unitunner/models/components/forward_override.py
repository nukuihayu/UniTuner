"""Own reversible instance forward overrides without changing computation."""

from collections.abc import Callable

from torch import nn


class ForwardOverride[R, **P]:
    """Install one callable and restore its predecessor in reverse ownership order.

    This object participates only in setup and cleanup, never in forward calls.
    Preserve whether the predecessor belonged to the instance or its class.
    """

    def __init__(self, module: nn.Module, replacement: Callable[P, R], *, owner: str) -> None:
        self._module = module
        self._replacement = replacement
        self._owner = owner
        self._had_forward = "forward" in module.__dict__
        self._original = module.forward
        self._closed = False
        module.__dict__["forward"] = replacement

    def check_current(self) -> None:
        """Reject out-of-order cleanup before an owner releases dependent resources."""
        if not self._closed and self._module.forward is not self._replacement:
            raise RuntimeError(f"{self._owner} must close after later forward owners")

    def close(self) -> None:
        """Restore the exact preceding override once; failed order checks remain retryable."""
        if self._closed:
            return
        self.check_current()
        if self._had_forward:
            self._module.__dict__["forward"] = self._original
        else:
            del self._module.__dict__["forward"]
        self._closed = True
