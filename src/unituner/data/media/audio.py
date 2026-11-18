"""Resolve audio sources and decode/resample their native timestamp-aligned waveforms."""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
from torch import Tensor

from unituner.config.audio import AudioDecodeConfig
from unituner.data.media.audio_timeline import DecodedAudio, assemble_audio_chunks
from unituner.data.media.video import read_video_origin

_AUDIO_EXTENSIONS = frozenset({".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"})
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioSource:
    """Identify a sidecar/explicit audio file or an embedded track sharing video PTS."""

    path: Path
    embedded: bool


def probe_audio(path: Path) -> bool:
    """Report whether the file contains an audio stream, closing the container."""
    with av.open(str(path)) as container:
        return bool(container.streams.audio)


def resolve_audio_source(video_path: Path, *, explicit_path: Path | None = None) -> AudioSource | None:
    """Resolve explicit audio, one same-stem sidecar, then an embedded audio track.

    Explicit files and sidecars must contain an audio stream. Multiple sidecars
    are ambiguous and fail rather than silently selecting a track. Explicit
    files have their own clock even when their path equals the video path.
    """
    video_path = video_path.resolve()
    if explicit_path is not None:
        path = explicit_path.resolve()
        if not path.is_file() or not probe_audio(path):
            raise ValueError(f"Explicit audio source requires an existing audio stream: {path}")
        return AudioSource(path, embedded=False)
    sidecars = sorted(
        (
            path.resolve()
            for path in video_path.parent.iterdir()
            if path.is_file() and path.stem == video_path.stem and path.suffix.lower() in _AUDIO_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    if len(sidecars) > 1:
        raise ValueError(f"Multiple same-stem audio sidecars for {video_path}")
    if sidecars:
        if not probe_audio(sidecars[0]):
            raise ValueError(f"Audio sidecar contains no audio stream: {sidecars[0]}")
        return AudioSource(sidecars[0], embedded=False)
    if video_path.is_file() and probe_audio(video_path):
        return AudioSource(video_path, embedded=True)
    return None


def _audio_chunk(frame: av.AudioFrame, channels: int) -> Tensor:
    """Copy resampled planar audio into independent CPU FP32 channels-by-samples storage."""
    array = frame.to_ndarray()
    if array.ndim != 2:
        raise ValueError("Audio resampler must return a planar matrix")
    if array.shape[0] != channels and array.shape[1] == channels:
        array = array.T
    if array.shape[0] != channels:
        raise ValueError("Audio resampler channel count differs from requested layout")
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())


def _chunk_start(frame: av.AudioFrame, sample_rate: int, fallback: int) -> int:
    if frame.pts is None or frame.time_base is None:
        return fallback
    return round(frame.pts * frame.time_base * sample_rate)


def decode_audio(source: AudioSource, config: AudioDecodeConfig) -> DecodedAudio:
    """Decode the first audio track to owned contiguous CPU FP32 CL at the chosen rate.

    PyAV converts channel layout and resamples to planar float, including the
    resampler flush. Preserve timestamp rounding and coarse container tick
    tolerance. Embedded tracks use the video's native first-frame origin;
    sidecars/explicit files start at their own first sample. Jitter, gaps and
    overlaps follow explicit timeline repair rules and return sample repair
    records. No input file or RNG changes; all containers close on failure.
    """
    channels, sample_rate = config.channels, config.sample_rate
    chunks: list[tuple[int, Tensor]] = []
    next_start = 0
    with av.open(str(source.path)) as container:
        if not container.streams.audio:
            raise ValueError(f"Audio source has no audio stream: {source.path}")
        stream = container.streams.audio[0]
        tolerance = 2
        if stream.time_base is not None:
            tolerance = max(tolerance, math.ceil(float(stream.time_base) * sample_rate) + 1)
        resampler = av.AudioResampler(format="fltp", layout="mono" if channels == 1 else "stereo", rate=sample_rate)
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunk = _audio_chunk(resampled, channels)
                start = _chunk_start(resampled, sample_rate, next_start)
                chunks.append((start, chunk))
                next_start = start + chunk.shape[1]
        for resampled in resampler.resample(None):
            chunk = _audio_chunk(resampled, channels)
            start = _chunk_start(resampled, sample_rate, next_start)
            chunks.append((start, chunk))
            next_start = start + chunk.shape[1]
    if not chunks:
        raise ValueError(f"Audio source decoded no samples: {source.path}")
    origin = read_video_origin(source.path) if source.embedded else None
    max_discontinuity = round(sample_rate * config.max_discontinuity_seconds)
    decoded = assemble_audio_chunks(
        chunks,
        channels=channels,
        timestamp_tolerance_samples=tolerance,
        pts_jitter_range_samples=max_discontinuity,
        max_gap_fill_samples=max_discontinuity,
        max_overlap_trim_samples=max_discontinuity,
        origin_sample=None if origin is None else round(origin * sample_rate),
        context=str(source.path),
    )
    repairs = decoded.repairs
    if origin is not None:
        offset = chunks[0][0] - round(origin * sample_rate)
        if offset > tolerance:
            repairs = repairs[1:]
        if abs(offset) > max_discontinuity:
            _LOGGER.info(
                "Audio starts %d samples %s the video origin: %s",
                abs(offset),
                "after" if offset > 0 else "before",
                source.path,
            )
    if repairs:
        _LOGGER.warning(
            "Audio timeline repaired %d gaps (%d samples) and %d overlaps (%d samples): %s",
            sum(r.filled > 0 for r in repairs),
            sum(r.filled for r in repairs),
            sum(r.trimmed > 0 for r in repairs),
            sum(r.trimmed for r in repairs),
            source.path,
        )
    return DecodedAudio(decoded.waveform.contiguous(), decoded.repairs)
