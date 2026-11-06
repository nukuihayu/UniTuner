"""Resolve declared component entry points only at their owning runtime boundary."""

from collections.abc import Callable
from importlib import import_module
from typing import Any


def load_provider(entry: str) -> Callable[..., Any]:
    """Import one explicit module:attribute entry and validate its callable contract."""
    module, separator, attribute = entry.partition(":")
    if not separator or not module or not attribute:
        raise ValueError(f"Invalid model provider entry: {entry}")
    provider = getattr(import_module(module), attribute)
    if not callable(provider):
        raise TypeError(f"Model provider is not callable: {entry}")
    return provider
