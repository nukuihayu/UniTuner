"""Resolve floating dtypes and compute contexts independently of model families."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

import torch
from torch import nn

from unituner.artifacts.formats.precision import FloatingPrecision

FLOAT_DTYPES: dict[FloatingPrecision, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

_SAMPLING_PRECISION: ContextVar[FloatingPrecision | None] = ContextVar("sampling_precision", default=None)


@contextmanager
def sampling_precision(precision: FloatingPrecision | Literal["auto"]) -> Generator[None]:
    """Scope preview compute policy without changing borrowed weights or global defaults."""
    token = _SAMPLING_PRECISION.set(None if precision == "auto" else precision)
    try:
        yield
    finally:
        _SAMPLING_PRECISION.reset(token)


def sampling_dtype(model: nn.Module) -> torch.dtype:
    """Resolve requested compute/state dtype independently of packed parameter storage."""
    precision = _SAMPLING_PRECISION.get()
    if precision is not None:
        return FLOAT_DTYPES[precision]
    dtype = getattr(model, "dtype", None)
    if isinstance(dtype, torch.dtype) and dtype in FLOAT_DTYPES.values():
        return dtype
    for module in model.modules():
        dtype = getattr(module, "compute_dtype", None)
        if isinstance(dtype, torch.dtype) and dtype in FLOAT_DTYPES.values():
            return dtype
    return next(
        (parameter.dtype for parameter in model.parameters() if parameter.dtype in (torch.float16, torch.bfloat16)),
        torch.float32,
    )


def sampling_context(model: nn.Module, device: torch.device, *, state_dtype: torch.dtype | None = None):
    """Use requested preview compute or resident floating storage, never packed weight codes.

    Solver state is independent of transformer compute. Preserve FP32
    standalone behavior under ambient autocast; previews explicitly select
    their training policy. Disable weight caching for staged/updated weights.
    """
    dtype = sampling_dtype(model)
    if _SAMPLING_PRECISION.get() is None and dtype == torch.float32 and state_dtype in (torch.bfloat16, torch.float16):
        dtype = state_dtype
    return torch.autocast(
        device.type,
        dtype=dtype if dtype != torch.float32 else torch.bfloat16,
        enabled=dtype != torch.float32,
        cache_enabled=False,
    )


def compute_context(device: torch.device, precision: FloatingPrecision):
    """Select autocast without caching weights whose storage can be rebound."""
    return torch.autocast(device.type, dtype=FLOAT_DTYPES[precision], enabled=precision != "fp32", cache_enabled=False)
