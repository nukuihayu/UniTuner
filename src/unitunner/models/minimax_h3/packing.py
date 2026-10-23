# Copyright 2025-2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Diffusers PR #14355 (abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc)
# via the pinned Musubi and old MixTuner implementations.
"""Pack and restore native video patches and channel-major stereo audio rows."""

import math
from dataclasses import dataclass

from torch import Tensor

from unitunner.models.minimax_h3.layout import (
    AUDIO_CHANNELS,
    STEREO_CHANNELS,
    VIDEO_CHANNELS,
    VIDEO_PATCH_SIZE,
    MiniMaxH3PackedLayout,
    MiniMaxH3VideoGeometry,
)


@dataclass(frozen=True, slots=True)
class MiniMaxH3Targets:
    """Own restored video B24FHW and stereo audio B32x2xA tensors (or unbatched)."""

    video: Tensor
    audio: Tensor


def pack_video_rows(latents: Tensor) -> Tensor:
    """Patch H3 video latents while preserving the leading batch axis."""
    if min(latents.shape, default=0) <= 0:
        raise ValueError("MiniMaxH3 packing requires nonempty latent axes")
    unbatched = latents.ndim == 4
    if unbatched:
        latents = latents.unsqueeze(0)
    if latents.ndim != 5:
        raise ValueError(f"MiniMax-H3 video latents must be [B,24,F,H,W], got {tuple(latents.shape)}")
    batch, channels, frames, height, width = latents.shape
    if channels != VIDEO_CHANNELS:
        raise ValueError(f"MiniMax-H3 video latents need {VIDEO_CHANNELS} channels, got {channels}")
    patch_t, patch_h, patch_w = VIDEO_PATCH_SIZE
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"MiniMax-H3 video latent shape {frames}x{height}x{width} is not divisible by patch {VIDEO_PATCH_SIZE}"
        )
    rows = latents.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    rows = rows.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
        batch,
        (frames // patch_t) * (height // patch_h) * (width // patch_w),
        channels * patch_t * patch_h * patch_w,
    )
    return rows[0] if unbatched else rows


def pack_audio_rows(latents: Tensor) -> Tensor:
    """Pack stereo H3 audio in channel-major order."""
    if min(latents.shape, default=0) <= 0:
        raise ValueError("MiniMaxH3 packing requires nonempty latent axes")
    unbatched = latents.ndim == 3
    if unbatched:
        latents = latents.unsqueeze(0)
    if latents.ndim != 4 or latents.shape[1] != AUDIO_CHANNELS or latents.shape[2] != STEREO_CHANNELS:
        raise ValueError(f"MiniMax-H3 audio latents must be [B,32,2,A], got {tuple(latents.shape)}")
    batch, channels, stereo, frames = latents.shape
    rows = latents.permute(0, 2, 3, 1).reshape(batch, stereo * frames, channels)
    return rows[0] if unbatched else rows


def _unpack_video_rows(rows: Tensor, video: MiniMaxH3VideoGeometry) -> Tensor:
    """Restore native video patches while preserving optional batch axes."""
    unbatched = rows.ndim == 2
    if unbatched:
        rows = rows.unsqueeze(0)
    expected_shape = (video.row_count, VIDEO_CHANNELS * math.prod(VIDEO_PATCH_SIZE))
    if rows.ndim != 3 or tuple(rows.shape[1:]) != expected_shape:
        raise ValueError(f"MiniMax-H3 target video rows must be [B,{expected_shape[0]},{expected_shape[1]}]")
    batch = rows.shape[0]
    patch_t, patch_h, patch_w = VIDEO_PATCH_SIZE
    unpacked = rows.reshape(
        batch,
        video.frames // patch_t,
        video.height // patch_h,
        video.width // patch_w,
        VIDEO_CHANNELS,
        patch_t,
        patch_h,
        patch_w,
    )
    unpacked = unpacked.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
        batch,
        VIDEO_CHANNELS,
        video.frames,
        video.height,
        video.width,
    )
    return unpacked[0] if unbatched else unpacked


def _unpack_audio_rows(rows: Tensor, audio_frames: int) -> Tensor:
    """Restore channel-major stereo audio while preserving optional batch axes."""
    unbatched = rows.ndim == 2
    if unbatched:
        rows = rows.unsqueeze(0)
    expected_shape = (audio_frames * STEREO_CHANNELS, AUDIO_CHANNELS)
    if rows.ndim != 3 or tuple(rows.shape[1:]) != expected_shape:
        raise ValueError(f"MiniMax-H3 target audio rows must be [B,{expected_shape[0]},{expected_shape[1]}]")
    batch = rows.shape[0]
    unpacked = rows.reshape(batch, STEREO_CHANNELS, audio_frames, AUDIO_CHANNELS).permute(0, 3, 1, 2)
    return unpacked[0] if unbatched else unpacked


def unpack_targets(
    layout: MiniMaxH3PackedLayout,
    target_video_rows: Tensor,
    target_audio_rows: Tensor,
) -> MiniMaxH3Targets:
    """Restore the two target streams with matching batch ownership."""
    if target_video_rows.ndim != target_audio_rows.ndim or target_video_rows.ndim not in {2, 3}:
        raise ValueError("MiniMax-H3 target video/audio rows must both be batched or both be unbatched")
    if target_video_rows.ndim == 3 and target_video_rows.shape[0] != target_audio_rows.shape[0]:
        raise ValueError("MiniMax-H3 target video/audio rows must have matching batch axes")
    return MiniMaxH3Targets(
        _unpack_video_rows(target_video_rows, layout.target_video),
        _unpack_audio_rows(target_audio_rows, layout.target_audio_frames),
    )
