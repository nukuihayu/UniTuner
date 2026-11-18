"""Verify bounded validation concurrency, deterministic order and error cleanup."""

from threading import Barrier, Lock

import pytest

from unituner.data.validation import ordered_validation_map


def test_validation_overlaps_work_and_retains_input_order():
    barrier = Barrier(2, timeout=5)
    lock = Lock()
    active = 0
    peak = 0

    def validate(value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait()
        with lock:
            active -= 1
        return value * 2

    assert list(ordered_validation_map(validate, range(20), workers=2)) == list(range(0, 40, 2))
    assert peak == 2


def test_validation_bounds_prefetch_for_large_inputs():
    consumed = []

    def inputs():
        for value in range(5000):
            consumed.append(value)
            yield value

    results = ordered_validation_map(lambda value: value, inputs(), workers=4)
    try:
        for expected in range(10):
            assert next(results) == expected
            assert len(consumed) <= expected + 8
    finally:
        results.close()
    assert len(consumed) < 5000


@pytest.mark.parametrize("workers", [1, 4])
def test_validation_propagates_failure_and_closes_pending_work(workers):
    def validate(value):
        if value == 2:
            raise ValueError("corrupt record")
        return value

    results = ordered_validation_map(validate, range(20), workers=workers)
    assert next(results) == 0
    assert next(results) == 1
    with pytest.raises(ValueError, match="corrupt record"):
        next(results)
    assert list(results) == []
