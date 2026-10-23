"""Runtime assertions for optional values produced by test fixtures."""

from typing import Any


def require_type[T](value: object, expected: type[T]) -> T:
    """Check the concrete type of a module fetched from a dynamic Torch container."""
    assert isinstance(value, expected)
    return value


def invalid_input(value: object) -> Any:
    """Pass deliberately invalid data through to the runtime validator under test."""
    return value


def require_value[T](value: T | None) -> T:
    """Assert presence inside comprehensions without discarding static types."""
    assert value is not None
    return value
