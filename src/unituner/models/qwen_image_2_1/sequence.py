# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Interleave native text/reference tokens and construct causal segment geometry."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

type ImageGrid = tuple[int, int, int]
type AttentionSegment = tuple[int, int, bool]
type AttentionSegments = tuple[tuple[AttentionSegment, ...], ...]


@dataclass(frozen=True, slots=True)
class JointSequence:
    """Own padded BLD tokens, BL3 IDs, complex rotary phases and target spans.

    Segments exclude right padding. Target offsets are CPU geometry; the BL
    boolean mask selects target-time versus shared zero-time conditioning rows.
    """

    tokens: Tensor
    position_ids: Tensor
    rotary: Tensor
    segments: AttentionSegments
    target_mask: Tensor
    target_offsets: tuple[int, ...]


def rotary_embedding(position_ids: Tensor, axes: tuple[int, int, int]) -> Tensor:
    """Return complex64 BL(D/2) phases from FP32 axis positions; retain gradients."""
    phases = []
    for index, width in enumerate(axes):
        inverse = 10000 ** (-torch.arange(0, width, 2, device=position_ids.device, dtype=torch.float32) / width)
        phases.append(position_ids[..., index, None].float() * inverse)
    phase = torch.cat(phases, dim=-1)
    return torch.polar(torch.ones_like(phase), phase)


def interleave_sequence(
    target: Tensor,
    text: Tensor,
    references: tuple[Tensor, ...],
    grids: tuple[tuple[ImageGrid, ...], ...],
    text_lengths: tuple[int, ...],
    image_slots: tuple[tuple[int, ...], ...],
    axes: tuple[int, int, int],
) -> JointSequence:
    """Interleave projected BND images and BLD text with target-last centered RoPE.

    Geometry is validated by the transformer before projection. Each reference
    insertion consumes zero text tokens and advances rotary time by max(H,W).
    Text IDs advance all three axes; image spatial IDs center relative to target
    parity. Returns new padded tensors without modifying inputs or drawing RNG.
    """
    rows, positions, segments, offsets = [], [], [], []
    for batch_index in range(target.shape[0]):
        parts, row_ids, runs = [], [], []
        cursor = position = length = 0
        target_height, target_width = grids[batch_index][-1][1:]
        for index, ((_, height, width), end, image) in enumerate(
            zip(
                grids[batch_index],
                (*image_slots[batch_index], text_lengths[batch_index]),
                (*references, target),
                strict=True,
            )
        ):
            if end > cursor:
                count = end - cursor
                parts.append(text[batch_index, cursor:end])
                row_ids.append(
                    torch.arange(position, position + count, device=target.device).float()[:, None].expand(-1, 3)
                )
                runs.append((length, length + count, True))
                position += count
                length += count
            if index == len(references):
                offsets.append(length)
            parts.append(image[batch_index])
            y = (
                torch.arange(height, device=target.device).float()
                - (height - height // 2)
                + 0.5 * (height % 2 - target_height % 2)
            )
            x = (
                torch.arange(width, device=target.device).float()
                - (width - width // 2)
                + 0.5 * (width % 2 - target_width % 2)
            )
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            row_ids.append(torch.stack((torch.full_like(yy, position), yy, xx), dim=-1).reshape(-1, 3))
            runs.append((length, length + height * width, False))
            position += max(height, width)
            length += height * width
            cursor = end
        rows.append(torch.cat(parts))
        positions.append(torch.cat(row_ids))
        segments.append(tuple(runs))
    max_length = max(row.shape[0] for row in rows)
    tokens = torch.stack([F.pad(row, (0, 0, 0, max_length - row.shape[0])) for row in rows])
    position_ids = torch.stack([F.pad(row, (0, 0, 0, max_length - row.shape[0])) for row in positions])
    indices = torch.arange(max_length, device=target.device)[None]
    starts = torch.tensor(offsets, device=target.device)[:, None]
    mask = (indices >= starts) & (indices < starts + target.shape[1])
    return JointSequence(
        tokens, position_ids, rotary_embedding(position_ids, axes), tuple(segments), mask, tuple(offsets)
    )
