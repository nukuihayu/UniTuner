"""Validate decoded H3 cache inputs independently of encoder construction."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from unituner.config.audio import AudioDecodeConfig
from unituner.data.media.audio import AudioSource
from unituner.models.minimax_h3.data.cache.storage import (
    MiniMaxH3CacheFormat,
    MiniMaxH3CacheItem,
    MiniMaxH3CacheReference,
)
from unituner.models.minimax_h3.data.sources import MiniMaxH3Record, validate_reference_policy
from unituner.models.minimax_h3.layout import audio_latent_frames, video_latent_frames


class MiniMaxH3CachePolicy(BaseModel):
    """Pin posterior seed, native reference waveform repair limits and duration opt-in."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    cache_seed: int = 0
    allow_experimental_duration: bool = False
    audio_decode: AudioDecodeConfig = Field(default_factory=lambda: AudioDecodeConfig(sample_rate=32000, channels=2))
    audio_pad_tolerance: int = Field(default=800, ge=0)
    audio_max_repair_samples: int = Field(default=6400, ge=0)

    @model_validator(mode="after")
    def validate_audio_clock(self) -> Self:
        """Keep the released32kHz stereo clock for both target and reference VAEs."""
        if self.audio_decode.sample_rate != 32000 or self.audio_decode.channels != 2:
            raise ValueError("MiniMaxH3 cache audio requires32kHz stereo")
        return self


@dataclass(frozen=True, slots=True)
class MiniMaxH3DecodedItem:
    """Borrow a target crop and optional one-frame FL controls from shared media traversal.

    Frames are THWC RGB/RGBA uint8 or floating[0,1]. Audio is aligned stereo2L
    FP32 at32kHz, including exact terminal padding. Missing target audio uses
    zero waveform samples, not zero VAE latents. Controls are ordered HWC
    images on the target canvas; indices are unsorted native24Hz coordinates.
    item_key selects the one-frame posterior seed; video uses path#start:count.
    """

    record: MiniMaxH3Record
    frames: Tensor
    waveform: Tensor
    audio_present: bool
    crop_start_frame: int = 0
    target_index: int | None = None
    control_frames: tuple[Tensor, ...] = ()
    control_indices: tuple[int, ...] = ()
    item_key: str | None = None
    audio_source: AudioSource | None = None


def build_cache_item(
    decoded: MiniMaxH3DecodedItem, format: MiniMaxH3CacheFormat, policy: MiniMaxH3CachePolicy
) -> MiniMaxH3CacheItem:
    """Reject invalid target/condition windows before any encoder or reference media IO."""
    frames = decoded.frames
    if frames.ndim != 4 or min(frames.shape) <= 0 or frames.shape[-1] not in (3, 4):
        raise ValueError("MiniMaxH3 target pixels require nonempty THWC RGB/RGBA")
    count, height, width = frames.shape[:3]
    if width % 32 or height % 32:
        raise ValueError("MiniMaxH3 target canvas must be32-aligned")
    single = format.frame_layout == "single-frame"
    validate_reference_policy(decoded.record.references, format.task, one_frame=single)
    if single:
        if count != 1 or decoded.target_index is None or decoded.audio_present or decoded.crop_start_frame:
            raise ValueError("MiniMaxH3 one-frame target requires one image, target_index, silent audio and no crop")
        audio_frames = 2
    else:
        video_latent_frames(count)
        audio_frames = audio_latent_frames(count)
        if (
            decoded.target_index is not None
            or decoded.control_frames
            or decoded.control_indices
            or decoded.item_key is not None
        ):
            raise ValueError("MiniMaxH3 explicit controls/indices/item_key require one-frame data")
        if not policy.allow_experimental_duration and not Fraction(5) <= Fraction(count, 24) <= Fraction(15):
            raise ValueError("MiniMaxH3 target duration must be5-15 seconds unless explicitly allowed")
    if (
        decoded.waveform.shape != (2, audio_frames * 800)
        or decoded.waveform.dtype != torch.float32
        or not torch.isfinite(decoded.waveform).all()
        or type(decoded.audio_present) is not bool
    ):
        raise ValueError("MiniMaxH3 target waveform requires finite FP32 stereo with exact synchronized length")
    if not decoded.audio_present and torch.count_nonzero(decoded.waveform):
        raise ValueError("MiniMaxH3 absent target audio requires a zero waveform")
    if decoded.item_key is not None and not decoded.item_key:
        raise ValueError("MiniMaxH3 one-frame item_key must be nonempty")
    if len(decoded.control_frames) != len(decoded.control_indices):
        raise ValueError("MiniMaxH3 one-frame control images require one index each")
    if decoded.control_frames and (not single or format.task != "fl2va"):
        raise ValueError("MiniMaxH3 control images require one-frame FL2VA")
    if single and format.task == "fl2va" and not decoded.control_frames:
        raise ValueError("MiniMaxH3 one-frame FL2VA requires control images")
    if any(
        control.ndim != 3 or control.shape[:2] != (height, width) or control.shape[-1] not in (3, 4)
        for control in decoded.control_frames
    ):
        raise ValueError("MiniMaxH3 control images require RGB/RGBA on the target spatial canvas")
    return MiniMaxH3CacheItem(
        frame_count=count,
        crop_start_frame=decoded.crop_start_frame,
        audio_present=decoded.audio_present,
        references=tuple(
            MiniMaxH3CacheReference(kind=ref.kind, has_audio=ref.kind == "video" and ref.audio is not None)
            for ref in decoded.record.references
        ),
        condition_roles=("first", "last") if format.task == "fl2va" and not single else (),
        target_index=decoded.target_index,
        control_indices=decoded.control_indices,
    )
