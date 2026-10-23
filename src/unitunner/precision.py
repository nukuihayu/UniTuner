"""Resolve floating dtypes and compute contexts independently of model families."""

import torch

from unitunner.artifacts.formats.precision import FloatingPrecision

FLOAT_DTYPES: dict[FloatingPrecision, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def compute_context(device: torch.device, precision: FloatingPrecision):
    """Select autocast without caching weights whose storage can be rebound."""
    return torch.autocast(device.type, dtype=FLOAT_DTYPES[precision], enabled=precision != "fp32", cache_enabled=False)
