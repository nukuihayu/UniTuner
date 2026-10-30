# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Compute Wan native FP64 time and complex rotary features without device readback."""

import torch
from torch import Tensor

type VideoGrid = tuple[int, int, int]


def timestep_embedding(timesteps: Tensor, width: int) -> Tensor:
    """Return FP64 cosine/sine features for flattened native scheduler times.

    Times use the model's scheduler units, usually 0-1000, not noise fractions.
    Preserve native FP64 outer-product arithmetic before the caller casts FP32.
    """
    half = width // 2
    positions = timesteps.to(torch.float64)
    angles = torch.outer(positions, torch.pow(10000, -torch.arange(half).to(positions).div(half)))
    return torch.cat((angles.cos(), angles.sin()), dim=1)


def rotary_embedding(grid: VideoGrid, head_dim: int, device: torch.device, frames: tuple[int, ...] | None) -> Tensor:
    """Return L1(D/2) complex128 native F/H/W rotations for host-owned geometry.

    Explicit temporal indices are integer coordinates in the native 1024-entry
    table. No persistent buffer can be accidentally cast to a real model dtype.
    """
    count, height, width = grid
    axes = (head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6))
    coordinates = (tuple(range(count)) if frames is None else frames, tuple(range(height)), tuple(range(width)))
    rotations = []
    for index, (axis, positions) in enumerate(zip(axes, coordinates, strict=True)):
        frequencies = 1.0 / torch.pow(10000, torch.arange(0, axis, 2, dtype=torch.float64, device=device).div(axis))
        angles = torch.outer(torch.tensor(positions, device=device), frequencies)
        values = torch.polar(torch.ones_like(angles), angles)
        shape = [1, 1, 1, axis // 2]
        shape[index] = len(positions)
        rotations.append(values.view(shape).expand(count, height, width, -1))
    return torch.cat(rotations, dim=-1).reshape(count * height * width, 1, head_dim // 2)


def rotate(values: Tensor, rotations: tuple[Tensor, ...]) -> Tensor:
    """Rotate valid BLHD prefixes in FP64, preserve padding and restore input dtype.

    Use an out-of-place expression so backward never depends on overwritten
    projection storage. Host-owned lengths require no GPU synchronization.
    """
    rows = []
    for row, frequencies in zip(values, rotations, strict=True):
        count = frequencies.shape[0]
        pairs = torch.view_as_complex(row[:count].to(torch.float64).reshape(count, row.shape[1], -1, 2))
        rotated = torch.view_as_real(pairs * frequencies).flatten(2).to(values.dtype)
        rows.append(torch.cat((rotated, row[count:])))
    return torch.stack(rows)
