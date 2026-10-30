"""Traverse H3 target media into owned native pixels and synchronized waveform windows."""

from collections.abc import Iterable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import Self

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from unituner.config.datasets import VideoDatasetConfig
from unituner.data.media.audio import decode_audio, resolve_audio_source
from unituner.data.media.audio_timeline import audio_window_start, slice_audio_window
from unituner.data.media.buckets import ImageBuckets
from unituner.data.media.image import resize_pixels
from unituner.data.media.video import read_video, select_video_windows, video_source_size
from unituner.models.minimax_h3.data.cache.inputs import MiniMaxH3CachePolicy, MiniMaxH3DecodedItem
from unituner.models.minimax_h3.data.cache.storage import MiniMaxH3CacheFormat
from unituner.models.minimax_h3.data.sources import MiniMaxH3Record, validate_reference_policy
from unituner.models.minimax_h3.layout import audio_latent_frames, video_latent_frames


class MiniMaxH3TraversalPolicy(BaseModel):
    """Pin explicit target canvas and24Hz timestamp windows on the17*n+5 grid.

    Resolution is width,height on the32-pixel grid. The default124-frame crop
    is within the released5-15second duration range. Short experimental crops
    require an independent explicit opt-in in MiniMaxH3CachePolicy.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    resolution: tuple[int, int]
    start_frame: int = Field(default=0, ge=0)
    end_frame: int | None = Field(default=None, gt=0)
    video: VideoDatasetConfig = Field(
        default_factory=lambda: VideoDatasetConfig(
            target_frames=(124,), max_frames=345, target_fps=24.0, fps_resampling="timestamps"
        )
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Reject ambiguous FPS and target counts instead of silently rounding them."""
        if self.end_frame is not None and self.end_frame <= self.start_frame:
            raise ValueError("MiniMaxH3 traversal requires ordered start_frame/end_frame")
        if any(value <= 0 or value % 32 for value in self.resolution):
            raise ValueError("MiniMaxH3 target resolution must be positive32-aligned width,height")
        if (
            self.video.target_fps != 24
            or self.video.fps_resampling != "timestamps"
            or self.video.source_fps is not None
        ):
            raise ValueError("MiniMaxH3 target video requires24Hz timestamp resampling without source_fps")
        for count in self.video.target_frames:
            video_latent_frames(count)
        if self.video.max_frames < 5:
            raise ValueError("MiniMaxH3 video.max_frames must allow at least5 frames")
        return self


def _read_pixels(path: Path, resolution: tuple[int, int]) -> Tensor:
    """Read owned CPU uint8 HWC RGB(A), retaining alpha through native bucket resize."""
    with Image.open(path) as decoded:
        with decoded.copy() if decoded.mode in ("RGB", "RGBA") else decoded.convert("RGB") as image:
            return torch.from_numpy(resize_pixels(np.array(image), resolution))


def traverse_minimax_h3(
    records: Iterable[MiniMaxH3Record],
    *,
    format: MiniMaxH3CacheFormat,
    traversal: MiniMaxH3TraversalPolicy,
    policy: MiniMaxH3CachePolicy,
    buckets: ImageBuckets | None = None,
) -> Iterator[MiniMaxH3DecodedItem]:
    """Yield native target windows without loading encoders or consuming global RNG.

    Image/control pixels retain RGB(A) through shared Lanczos
    center cropping. Video uses shared nearest-PTS24Hz sampling and the same
    spatial resize. Decode target video/audio once per record; each yielded
    tensor owns storage, including overlapping/repeated windows. Embedded
    audio shares video origin; explicit/sidecar audio has its own origin.
    Missing/disabled audio yields exact zeros before VAE encoding.

    Omitted target_audio_path selects unique same-stem sidecar then embedded
    audio. Explicit null disables audio. Single-frame targets always use1600
    silence samples and keep ordered, unsorted or repeated control indices.
    All media resources close before yielding; invalid records fail explicitly.
    """
    single = format.frame_layout == "single-frame"
    for record in records:
        resolution = (
            traversal.resolution
            if buckets is None
            else (
                buckets.for_image(record.target_path)
                if format.frame_layout == "single-frame"
                else buckets.select(video_source_size(record.target_path))
            )
        )
        validate_reference_policy(record.references, format.task, one_frame=single)
        if record.controls and (not single or format.task != "fl2va"):
            raise ValueError("MiniMaxH3 controls require one-frame FL2VA")
        if single:
            if traversal.start_frame or traversal.end_frame is not None:
                raise ValueError("MiniMaxH3 one-frame targets do not accept video bounds")
            if record.target_audio_path is not None:
                raise ValueError("MiniMaxH3 one-frame targets require silent audio")
            if format.task == "fl2va" and not record.controls:
                raise ValueError("MiniMaxH3 one-frame FL2VA requires controls")
            frames = _read_pixels(record.target_path, resolution).unsqueeze(0)
            controls = tuple(_read_pixels(control.path, resolution) for control in record.controls)
            yield MiniMaxH3DecodedItem(
                record,
                frames,
                torch.zeros(2, 1600, dtype=torch.float32),
                False,
                target_index=0 if record.target_index is None else record.target_index,
                control_frames=controls,
                control_indices=tuple(control.index for control in record.controls),
            )
            continue
        if record.target_index is not None:
            raise ValueError("MiniMaxH3 target_index requires one-frame targets")
        if not policy.allow_experimental_duration and traversal.video.extraction != "full":
            if any(not Fraction(5) <= Fraction(count, 24) <= Fraction(15) for count in traversal.video.target_frames):
                raise ValueError("MiniMaxH3 target duration must be5-15 seconds unless explicitly allowed")
        source = (
            resolve_audio_source(record.target_path, explicit_path=record.target_audio_path)
            if record.target_audio_enabled
            else None
        )
        video = read_video(
            record.target_path,
            resolution,
            traversal.video,
            start_frame=traversal.start_frame,
            end_frame=traversal.end_frame,
        )
        windows = select_video_windows(len(video.pixels), traversal.video, temporal_factor=17, temporal_offset=5)
        if not policy.allow_experimental_duration and any(
            not Fraction(5) <= Fraction(window.count, 24) <= Fraction(15) for window in windows
        ):
            raise ValueError("MiniMaxH3 target duration must be5-15 seconds unless explicitly allowed")
        audio = None if source is None else decode_audio(source, policy.audio_decode)
        for window in windows:
            samples = audio_latent_frames(window.count) * 800
            waveform = (
                torch.zeros(2, samples, dtype=torch.float32)
                if audio is None
                else slice_audio_window(
                    audio.waveform,
                    start_sample=audio_window_start(traversal.start_frame + window.start, 24, 32000),
                    sample_count=samples,
                    pad_tolerance=policy.audio_pad_tolerance,
                    repairs=audio.repairs,
                    max_repair_samples=policy.audio_max_repair_samples,
                    context=str(record.target_path),
                ).clone()
            )
            yield MiniMaxH3DecodedItem(
                record,
                torch.from_numpy(video.pixels[window.start : window.start + window.count].copy()),
                waveform,
                source is not None,
                crop_start_frame=traversal.start_frame + window.start,
                audio_source=source,
            )
