# Copyright 2025-2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Diffusers PR #14355 (abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc)
# via the pinned Musubi and old MixTuner implementations.
"""Map native clean-fraction AV times and modality tags to AdaLN row selections."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from unituner.models.minimax_h3.layout import MiniMaxH3PackedLayout


@dataclass(frozen=True, slots=True)
class MiniMaxH3TimestepRows:
    """Own CPU clean-fraction times, modality tags and contiguous AdaLN selections."""

    unique_timesteps: Tensor
    row_timesteps: Tensor
    row_timestep_indices: Tensor
    token_tags: Tensor
    block_adaln_indices: Tensor
    block_segments: tuple[tuple[int, int, int], ...]
    video_timestep_index: int
    audio_timestep_index: int


def _single_model_time(value: float | Tensor, label: str) -> float:
    """Validate one native clean-fraction scalar, synchronizing only at this boundary."""
    tensor = torch.as_tensor(value).detach().reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"MiniMax-H3 R1 {label} must contain exactly one value")
    scalar = float(tensor[0].item())
    if not math.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
        raise ValueError(f"MiniMax-H3 {label} must be finite and in [0,1], got {scalar}")
    return scalar


def validate_clean_coefficient(value: float, label: str) -> float:
    """The condition-augmentation clean coefficient (clean*x + (1-clean)*eps) must lie in [0,1]; the timestep rows, the samplers and the CLIs all validate through here."""
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"MiniMax-H3 {label} must be finite and in [0,1], got {value}")
    return value


def _build_modulation_segments(indices: Tensor) -> tuple[tuple[int, int, int], ...]:
    """Group contiguous CPU row selections into half-open modulation runs."""
    change_points = torch.nonzero(indices[1:] != indices[:-1], as_tuple=False).flatten() + 1
    boundaries = torch.cat((indices.new_tensor([0]), change_points, indices.new_tensor([indices.numel()])))
    return tuple(
        tuple(row)
        for row in torch.stack(
            (boundaries[:-1], boundaries[1:], indices[boundaries[:-1]]),
            dim=1,
        ).tolist()
    )


def build_timestep_rows(
    layout: MiniMaxH3PackedLayout,
    *,
    text_token_tags: Tensor,
    model_t_video: float | Tensor,
    model_t_audio: float | Tensor,
    visual_condition_clean: float = 0.999,
    audio_condition_clean: float = 1.0,
) -> MiniMaxH3TimestepRows:
    """Build CPU AdaLN rows with native0=noise,1=clean times independently per modality."""
    video_time = _single_model_time(model_t_video, "video model time")
    audio_time = _single_model_time(model_t_audio, "audio model time")
    visual_clean = validate_clean_coefficient(visual_condition_clean, "visual condition clean coefficient")
    audio_clean = validate_clean_coefficient(audio_condition_clean, "audio condition clean coefficient")

    text_token_tags = torch.as_tensor(text_token_tags)
    if text_token_tags.dtype != torch.int64 or text_token_tags.shape != (1, layout.text_length):
        raise ValueError(f"MiniMax-H3 text token tags must be int64 [1,{layout.text_length}]")
    if not torch.all((text_token_tags == 0) | (text_token_tags == 1)):
        raise ValueError("MiniMax-H3 text token tags may contain only 0 and 1")
    text_token_tags = text_token_tags.detach().cpu()

    segment_times = {}
    for segment in layout.segments:
        if segment.kind in {"text", "target_video"}:
            segment_times[segment.role] = video_time
        elif segment.kind == "target_audio":
            segment_times[segment.role] = audio_time
        elif segment.kind == "visual_condition":
            segment_times[segment.role] = max(video_time, visual_clean)
        else:
            segment_times[segment.role] = max(audio_time, audio_clean)

    unique_values = sorted(set(segment_times.values()))
    timestep_index = {value: index for index, value in enumerate(unique_values)}
    row_timesteps = torch.empty((1, layout.row_count), dtype=torch.float32)
    row_timestep_indices = torch.empty((1, layout.row_count), dtype=torch.int64)
    token_tags = torch.empty((1, layout.row_count), dtype=torch.int64)
    for segment in layout.segments:
        value = segment_times[segment.role]
        row_timesteps[:, segment.row_slice] = value
        row_timestep_indices[:, segment.row_slice] = timestep_index[value]
        if segment.kind == "text":
            token_tags[:, segment.row_slice] = text_token_tags
        elif segment.kind in {"visual_condition", "target_video"}:
            token_tags[:, segment.row_slice] = 0
        else:
            token_tags[:, segment.row_slice] = 2

    block_adaln_indices = 3 * row_timestep_indices + token_tags
    block_segments = _build_modulation_segments(block_adaln_indices[0])

    return MiniMaxH3TimestepRows(
        unique_timesteps=torch.tensor(unique_values, dtype=torch.float32),
        row_timesteps=row_timesteps,
        row_timestep_indices=row_timestep_indices,
        token_tags=token_tags,
        block_adaln_indices=block_adaln_indices,
        block_segments=block_segments,
        video_timestep_index=timestep_index[video_time],
        audio_timestep_index=timestep_index[audio_time],
    )
