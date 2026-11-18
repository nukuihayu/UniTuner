# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Decode owned video frame grids and deterministic native temporal windows."""

import logging
import math
from bisect import bisect_left
from collections.abc import Generator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from unituner.artifacts.hashing import canonical_sha256, file_sha256
from unituner.config.datasets import VideoDatasetConfig
from unituner.data.media.image import discover_images, resize_pixels
from unituner.data.media.probe import video_size

_EXTENSIONS = frozenset({".mp4", ".webm", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg"})
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VideoFrames:
    """Own CPU uint8 THWC RGB pixels and their original decoded-frame indices."""

    pixels: NDArray[np.uint8]
    source_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VideoWindow:
    """Select a contiguous output-frame-grid window; start is inclusive."""

    start: int
    count: int


@dataclass(frozen=True, slots=True)
class TimestampVideoFrames:
    """Carry uncropped RGB frames sampled onto a fixed clock and their source indices.

    Repeated source indices share their decoded array. Consumers must treat the
    frames as borrowed; per-frame resizing can accommodate changing source sizes.
    """

    frames: tuple[NDArray[np.uint8], ...]
    source_indices: tuple[int, ...]


def discover_videos(directory: Path) -> tuple[Path, ...]:
    """Return sorted immediate video files or frame directories, with unique stems.

    Frame directories must contain supported images. Ignore nonmedia/cache
    subdirectories so a cache inside the source does not become another clip.
    """
    candidates = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in _EXTENSIONS:
            candidates.append(path)
        elif path.is_dir():
            try:
                discover_images(path)
            except ValueError as error:
                from unituner.logging import get_logger

                get_logger("data.video").debug("Skipping non-frame directory %s: %s", path, error)
                continue
            candidates.append(path)
    paths = tuple(candidates)
    if not paths or len({path.stem for path in paths}) != len(paths):
        raise ValueError("Video source requires nonempty media with unique stems")
    return paths


def video_source_size(path: Path) -> tuple[int, int]:
    """Read native pixel (width,height) from a container or the first directory frame."""
    if path.is_dir():
        with Image.open(discover_images(path)[0]) as image:
            return image.size
    return video_size(path)


def video_source_sha256(path: Path) -> str:
    """Hash file bytes or the ordered relative frame filenames and their complete bytes."""
    return (
        canonical_sha256({frame.name: file_sha256(frame) for frame in discover_images(path)})
        if path.is_dir()
        else file_sha256(path)
    )


def resample_frame_indices(
    timestamps: tuple[float, ...], *, source_frame_duration: float, target_fps: float
) -> tuple[int, ...]:
    """Map source PTS seconds to nearest frames on a fixed target-FPS grid.

    Preserve native earlier-frame tie breaking and the final-frame duration.
    Stable-sort small timestamp reorderings, but reject backward steps larger
    than four source frame durations. Returned indices refer to decode order.
    """
    if any(not math.isfinite(value) or value <= 0 for value in (source_frame_duration, target_fps)):
        raise ValueError("Video frame duration and target FPS must be finite and positive")
    if any(not math.isfinite(value) for value in timestamps):
        raise ValueError("Video timestamps must be finite")
    if not timestamps:
        return ()
    backwards = [index for index in range(1, len(timestamps)) if timestamps[index] < timestamps[index - 1]]
    if any(timestamps[index - 1] - timestamps[index] > 4 * source_frame_duration for index in backwards):
        raise ValueError("Video timestamps must be nondecreasing within four source frames")
    order = list(range(len(timestamps)))
    if backwards:
        _LOGGER.warning("Reordering %d backward video timestamp steps within four source frames", len(backwards))
        order.sort(key=timestamps.__getitem__)
    origin = timestamps[order[0]]
    normalized = [timestamps[index] - origin for index in order]
    count = max(1, math.ceil((normalized[-1] + source_frame_duration) * target_fps - 1e-9))
    result = []
    for index in range(count):
        target = index / target_fps
        right = bisect_left(normalized, target)
        if right == 0:
            selected = 0
        elif right == len(normalized):
            selected = right - 1
        else:
            selected = right - 1 if target - normalized[right - 1] <= normalized[right] - target + 1e-12 else right
        result.append(order[selected])
    return tuple(result)


def _decode_frames(path: Path, target_fps: float) -> Generator[tuple[NDArray[np.uint8], float, float]]:
    """Own decoder lifetime; yield RGB, PTS seconds and estimated source frame duration."""
    if path.is_dir():
        for index, image_path in enumerate(discover_images(path)):
            with Image.open(image_path) as image, image.convert("RGB") as rgb:
                yield np.array(rgb), index / target_fps, 1 / target_fps
    else:
        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                raise ValueError("Video source has no video stream")
            stream = container.streams.video[0]
            rate = float(stream.average_rate) if stream.average_rate is not None else target_fps
            duration = 1 / rate if rate > 0 else 1 / target_fps
            for index, frame in enumerate(container.decode(stream)):
                timestamp = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None and frame.time_base is not None
                    else index * duration
                )
                yield np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8), timestamp, duration


def read_video_origin(path: Path) -> float | None:
    """Read the earliest PTS among the first nine decoded video frames, in seconds.

    Match the native four-frame timestamp-reorder policy's origin probe. Return
    None without video frames, or0 if a probed frame has no timestamp. The decoder
    closes on normal completion, an early return or failure.
    """
    origin = None
    with av.open(str(path)) as container:
        if not container.streams.video:
            return None
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if frame.pts is None or frame.time_base is None:
                return 0.0
            timestamp = float(frame.pts * frame.time_base)
            origin = timestamp if origin is None else min(origin, timestamp)
            if index >= 8:
                break
    return origin


def read_timestamp_video(
    path: Path, *, target_fps: float, start_frame: int = 0, end_frame: int | None = None
) -> TimestampVideoFrames:
    """Decode native RGB pixels onto a fixed timestamp grid without cropping/resizing.

    Use nearest-frame selection, earlier-frame ties and the final-frame duration.
    Slice indices refer to the resampled grid. Preserve original source frame
    sizes so family-specific reference policies can resize each independently.
    Frame storage outlives the closed decoder; repeated selections share arrays.
    """
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("Timestamp video target_fps must be finite and positive")
    if (
        type(start_frame) is not int
        or start_frame < 0
        or (end_frame is not None and (type(end_frame) is not int or end_frame <= start_frame))
    ):
        raise ValueError("Timestamp video requires ordered nonnegative frame bounds")
    if not path.is_file():
        raise ValueError("Timestamp video resampling requires a video file")
    raw, timestamps, duration = [], [], 1 / target_fps
    with closing(_decode_frames(path, target_fps)) as decoded:
        for pixels, timestamp, frame_duration in decoded:
            raw.append(pixels)
            timestamps.append(timestamp)
            duration = frame_duration
    indices = resample_frame_indices(tuple(timestamps), source_frame_duration=duration, target_fps=target_fps)[
        start_frame:end_frame
    ]
    if not indices:
        raise ValueError("Timestamp video decoded no selected frames")
    return TimestampVideoFrames(tuple(raw[index] for index in indices), indices)


def read_video(
    path: Path,
    resolution: tuple[int, int],
    policy: VideoDatasetConfig,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> VideoFrames:
    """Decode/resample/crop RGB pixels without implicit FPS conversion or RNG.

    Fractional mode with no source_fps preserves every decoded frame; with a
    source_fps it follows native cumulative fractional dropping without frame
    duplication, including upsampling gaps. Timestamp mode selects nearest PTS
    frames and can duplicate frames; it requires a video file. Slice indices
    address the resampled grid. Every decoder/image closes on failure or early
    termination; returned pixels own contiguous THWC uint8 storage.
    """
    if min(resolution) < 1 or start_frame < 0 or (end_frame is not None and end_frame <= start_frame):
        raise ValueError("Video requires positive resolution and ordered nonnegative frame bounds")
    if policy.fps_resampling == "timestamps" and not path.is_file():
        raise ValueError("Timestamp video resampling requires a video file")
    frames, selected = [], []
    with closing(_decode_frames(path, policy.target_fps)) as decoded:
        if policy.fps_resampling == "timestamps":
            raw, timestamps, duration = [], [], 1 / policy.target_fps
            for pixels, timestamp, frame_duration in decoded:
                raw.append(pixels)
                timestamps.append(timestamp)
                duration = frame_duration
            indices = resample_frame_indices(
                tuple(timestamps), source_frame_duration=duration, target_fps=policy.target_fps
            )[start_frame:end_frame]
            for index in indices:
                frames.append(resize_pixels(raw[index], resolution))
                selected.append(index)
        else:
            delta = None if policy.source_fps is None else policy.target_fps / policy.source_fps
            fractional, previous = 0.0, -1
            for index, (pixels, _, _) in enumerate(decoded):
                if delta is None:
                    target_index = index
                else:
                    target_index = int(fractional)
                    fractional += delta
                    if target_index == previous:
                        continue
                    previous = target_index
                if target_index < start_frame:
                    continue
                if end_frame is not None and target_index >= end_frame:
                    break
                frames.append(resize_pixels(pixels, resolution))
                selected.append(index)
    if not frames:
        raise ValueError("Video source decoded no selected frames")
    return VideoFrames(np.stack(frames), tuple(selected))


def select_video_windows(
    frame_count: int, policy: VideoDatasetConfig, *, temporal_factor: int = 4, temporal_offset: int = 1
) -> tuple[VideoWindow, ...]:
    """Select ordered head/chunk/slide/uniform/full windows on the output frame grid.

    Explicit target counts must be temporal_offset+k*temporal_factor. Full mode rounds only
    the available source tail down to that grid. Uniform preserves repeated
    sample positions when the requested sample count exceeds available starts.
    """
    if (
        type(temporal_factor) is not int
        or type(temporal_offset) is not int
        or temporal_factor < 1
        or temporal_offset < 1
        or frame_count < temporal_offset
        or any(value < temporal_offset or (value - temporal_offset) % temporal_factor for value in policy.target_frames)
    ):
        raise ValueError("Video window targets must follow temporal_offset+k*temporal_factor on a sufficient source")
    if policy.extraction == "full":
        count = min(frame_count, policy.max_frames)
        if count < temporal_offset:
            raise ValueError("Video max_frames is below the minimum temporal window")
        return (VideoWindow(0, (count - temporal_offset) // temporal_factor * temporal_factor + temporal_offset),)
    result = []
    for count in sorted(set(policy.target_frames)):
        if count > frame_count:
            continue
        match policy.extraction:
            case "head":
                starts = [0]
            case "chunk":
                starts = range(0, frame_count - count + 1, count)
            case "slide":
                starts = range(0, frame_count - count + 1, policy.stride)
            case "uniform":
                starts = np.linspace(0, frame_count - count, policy.samples, dtype=int).tolist()
        result.extend(VideoWindow(int(start), count) for start in starts)
    if not result:
        raise ValueError("Video source is shorter than every requested target window")
    return tuple(result)
