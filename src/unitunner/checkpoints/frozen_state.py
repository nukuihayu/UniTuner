"""Identify immutable parameter references without omitting mutable buffers or optimizer state."""

import hashlib

import torch
from torch import Tensor


def frozen_tensor_sha256(value: Tensor) -> str:
    """Hash exact dtype, shape and contiguous CPU bytes with bounded extra host storage."""
    data = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{data.dtype}:{tuple(data.shape)}:".encode())
    digest.update(memoryview(data.reshape(-1).view(dtype=torch.uint8).numpy()))
    return digest.hexdigest()
