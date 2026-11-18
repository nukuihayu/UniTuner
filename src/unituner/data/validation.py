"""Bound parallel dataset IO while retaining deterministic result order."""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import islice


def ordered_validation_map[T, R](function: Callable[[T], R], items: Iterable[T], *, workers: int) -> Iterator[R]:
    """Overlap IO with at most two pending items per worker and no tensor prefetch.

    Python's unbounded executor.map can queue an entire large dataset. This
    sliding window preserves input order for resume hashes and pair indices.
    Exceptions propagate at their ordered position; pending work is cancelled
    when validation fails or the consumer closes the iterator.
    """
    if workers < 1:
        raise ValueError("Dataset validation workers must be positive")
    if workers == 1:
        yield from map(function, items)
        return
    source = iter(items)
    pending: deque[Future[R]] = deque()
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dataset-validation")
    try:
        pending.extend(executor.submit(function, item) for item in islice(source, 2 * workers))
        while pending:
            yield pending.popleft().result()
            for item in islice(source, 1):
                pending.append(executor.submit(function, item))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
