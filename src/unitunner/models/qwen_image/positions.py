"""Build Qwen Image complex rotary positions without mutable device caches."""

import torch
from torch import Tensor

type ImageGrid = tuple[int, int, int]


def _frequencies(positions: Tensor, width: int) -> Tensor:
    """Evaluate native CPU FP32 polar frequencies, independent of default dtype."""
    inverse = 1.0 / torch.pow(10000, torch.arange(0, width, 2, dtype=torch.float32, device="cpu").div(width))
    angles = torch.outer(positions, inverse)
    return torch.polar(torch.ones_like(angles), angles)


def build_rotary(
    grids: tuple[ImageGrid, ...],
    text_length: int,
    axes: tuple[int, int, int],
    *,
    layered: bool,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return complex64 image/text rotations in native row-major grid order.

    Image spatial positions are centered with negative positions first. Ordinary
    ordered references start their first axis at their grid index; layered's last
    condition uses -1 and requires one frame. Text starts at the largest spatial
    half-extent, or layer count when larger. Preserve the reference's 4096-table
    bounds explicitly. CPU frequency evaluation matches its original host table,
    then transfers only used values; no model state, RNG or mode is changed.
    """
    if not grids or text_length < 1:
        raise ValueError("Qwen Image requires image grids and a positive text length")
    image = []
    offset = 0
    for index, (frames, height, width) in enumerate(grids):
        if min(frames, height, width) < 1 or index + frames > 4096 or max(height, width) > 8192:
            raise ValueError("Qwen Image grid exceeds positive native rotary bounds")
        condition = layered and index == len(grids) - 1
        if condition and frames != 1:
            raise ValueError("Qwen Image layered condition grid requires one frame")
        frame_positions = (
            torch.full((1,), -1, dtype=torch.int64, device="cpu")
            if condition
            else torch.arange(index, index + frames, dtype=torch.int64, device="cpu")
        )
        row_positions = torch.arange(-(height - height // 2), height // 2, dtype=torch.int64, device="cpu")
        column_positions = torch.arange(-(width - width // 2), width // 2, dtype=torch.int64, device="cpu")
        frame = _frequencies(frame_positions, axes[0])[:, None, None].expand(frames, height, width, -1)
        row = _frequencies(row_positions, axes[1])[None, :, None].expand(frames, height, width, -1)
        column = _frequencies(column_positions, axes[2])[None, None, :].expand(frames, height, width, -1)
        image.append(torch.cat((frame, row, column), dim=-1).reshape(frames * height * width, -1))
        offset = max(offset, height // 2, width // 2)
    if layered:
        offset = max(offset, len(grids) - 1)
    if offset + text_length > 4096:
        raise ValueError("Qwen Image text positions exceed native rotary bound4096")
    positions = torch.arange(offset, offset + text_length, dtype=torch.int64, device="cpu")
    text = torch.cat([_frequencies(positions, axis) for axis in axes], dim=-1)
    return torch.cat(image).to(device), text.to(device)


def apply_rotary(values: Tensor, frequencies: Tensor) -> Tensor:
    """Rotate BSHD query/key pairs in FP32 complex arithmetic, returning input dtype."""
    pairs = torch.view_as_complex(values.float().reshape(*values.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * frequencies.unsqueeze(1)).flatten(3).to(values.dtype)
