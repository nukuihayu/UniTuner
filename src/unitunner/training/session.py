"""Own registered training resources with deterministic reverse-order cleanup."""

from contextlib import ExitStack
from types import TracebackType
from typing import Protocol, Self


class Closable(Protocol):
    """Release an explicitly registered resource idempotently."""

    def close(self) -> None:
        """Complete owned work and release only this resource's mutations."""
        ...


class TrainingSession:
    """Register each installation immediately; close all owners despite cleanup errors.

    Context exit rolls back already-registered installations on setup failure.
    Producers must drain their own asynchronous work before restoring callables.
    This owner neither discovers resources nor globally synchronizes devices.
    """

    def __init__(self) -> None:
        self._resources: ExitStack = ExitStack()
        self._closed: bool = False

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("TrainingSession is closed")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    def own[R: Closable](self, resource: R) -> R:
        """Register a successfully constructed owner before constructing another."""
        if self._closed:
            resource.close()
            raise RuntimeError("TrainingSession is closed")
        self._resources.callback(resource.close)
        return resource

    def close(self) -> None:
        """Release all owners in reverse registration order, at most once."""
        if not self._closed:
            self._closed = True
            self._resources.close()
