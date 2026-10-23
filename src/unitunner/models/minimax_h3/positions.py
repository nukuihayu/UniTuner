# Copyright 2025-2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Diffusers PR #14355 (abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc)
# via the pinned Musubi and old MixTuner implementations.
"""Build native FP64 text, reference and target audio/video rotary coordinates."""

import math

import torch
from torch import Tensor

from unitunner.models.minimax_h3.layout import (
    FRAME_PER_TOKEN,
    FRAME_RESCALE,
    STEREO_CHANNELS,
    MiniMaxH3PackedLayout,
    MiniMaxH3VideoGeometry,
    reference_condition_role,
)


def _axis_from_sqrt_area(dimension: int, sqrt_area: float) -> Tensor:
    """Construct native FP64 aspect-normalized spatial patch coordinates."""
    ratio = dimension / sqrt_area
    count = dimension // 2
    return (torch.arange(count, dtype=torch.float64) * (ratio / count) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(video: MiniMaxH3VideoGeometry) -> tuple[Tensor, Tensor]:
    """Construct flattened spatial coordinates and the stereo audio width anchors."""
    sqrt_area = math.sqrt(video.height * video.width)
    height_axis = _axis_from_sqrt_area(video.height, sqrt_area)
    width_axis = _axis_from_sqrt_area(video.width, sqrt_area)
    height, width = torch.meshgrid(height_axis, width_axis, indexing="ij")
    return torch.stack((height.reshape(-1), width.reshape(-1)), dim=-1), width_axis


def _video_t_spans(frames: int, temporal_stretch: float = 1.0) -> list[float]:
    """Return native1,4,4,4,4 latent spans in40Hz rotary units."""
    return [temporal_stretch * FRAME_RESCALE * FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)] for index in range(frames)]


def _video_grid(video: MiniMaxH3VideoGeometry, cursor: float, temporal_stretch: float = 1.0) -> Tensor:
    """Construct flattened FP64 visual time/height/width coordinates."""
    spans = torch.tensor(_video_t_spans(video.frames, temporal_stretch), dtype=torch.float64)
    times = cursor + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))
    frame, _ = _frame_grid(video)
    grid = torch.empty(video.frames, frame.shape[0], 3, dtype=torch.float64)
    grid[..., 0] = times[:, None]
    grid[..., 1:] = frame[None]
    return grid.reshape(-1, 3)


def _audio_grid(cursor: float, frames: int, width_low: float, width_high: float) -> Tensor:
    """Construct channel-major stereo coordinates with distinct width anchors."""
    grid = torch.zeros(frames * STEREO_CHANNELS, 3, dtype=torch.float64)
    grid[:, 0] = (cursor + torch.arange(frames, dtype=torch.float64)).repeat(STEREO_CHANNELS)
    grid[:frames, 2] = width_low
    grid[frames:, 2] = width_high
    return grid


def build_position_grid(layout: MiniMaxH3PackedLayout, *, device: torch.device | str | None = None) -> Tensor:
    """Return1xNx3 FP64 coordinates; construct on CPU then optionally copy to device."""
    positions = torch.empty(layout.row_count, 3, dtype=torch.float64)
    text = layout.segment("text")
    positions[text.row_slice] = 0
    positions[text.row_slice, 0] = torch.arange(layout.text_length, dtype=torch.float64)

    target_frame, target_width = _frame_grid(layout.target_video)
    target_width_endpoints = float(target_width[0]), float(target_width[-1])
    cursor = float(layout.text_length)

    if layout.task == "fl2va":
        condition_segments = tuple(segment for segment in layout.segments if segment.kind == "visual_condition")
        if layout.time_overrides is not None:
            condition_times = [cursor + time for time in layout.time_overrides.condition_times]
        else:
            # the last-frame anchor sits on the final pixel frame of the target timeline,
            # so both the total span and the one-frame back-off scale with the stretch
            last_time = cursor + sum(_video_t_spans(layout.target_video.frames, layout.temporal_stretch))
            last_time -= FRAME_RESCALE * layout.temporal_stretch
            condition_times = [cursor if segment.role == "first" else last_time for segment in condition_segments]
        for segment, time in zip(condition_segments, condition_times, strict=True):
            positions[segment.row_slice, 0] = time
            positions[segment.row_slice, 1:] = target_frame
    elif layout.task == "ref2va":
        for index, reference in enumerate(layout.references):
            if reference.kind == "image":
                segment = layout.segment(reference_condition_role(index, "image"))
                frame, _ = _frame_grid(reference.visual_geometry)
                positions[segment.row_slice, 0] = cursor
                positions[segment.row_slice, 1:] = frame
                cursor += 1.0
            elif reference.kind == "audio":
                segment = layout.segment(reference_condition_role(index, "audio"))
                positions[segment.row_slice] = _audio_grid(cursor, reference.audio_frames, *target_width_endpoints)
                cursor += float(reference.audio_frames)
            else:
                frame, width = _frame_grid(reference.visual_geometry)
                if reference.audio_frames:
                    audio = layout.segment(reference_condition_role(index, "audio"))
                    positions[audio.row_slice] = _audio_grid(
                        cursor,
                        reference.audio_frames,
                        float(width[0]),
                        float(width[-1]),
                    )
                video = layout.segment(reference_condition_role(index, "video"))
                positions[video.row_slice] = _video_grid(reference.visual_geometry, cursor)
                cursor += max(float(reference.audio_frames), sum(_video_t_spans(reference.visual_geometry.frames)))

    if layout.time_overrides is not None:
        cursor += layout.time_overrides.target_time
    target_audio = layout.target_audio_segment
    positions[target_audio.row_slice] = _audio_grid(
        cursor,
        layout.target_audio_frames,
        *target_width_endpoints,
    )
    target_video = layout.target_video_segment
    positions[target_video.row_slice] = _video_grid(layout.target_video, cursor, layout.temporal_stretch)
    positions = positions.unsqueeze(0)
    return positions.to(device=device) if device is not None else positions
