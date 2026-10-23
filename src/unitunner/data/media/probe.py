"""Inspect media stream presence and duration without retaining decoder resources."""

from dataclasses import dataclass
from pathlib import Path

import av


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """Describe stream presence and the longest declared stream duration in seconds."""

    has_video: bool
    has_audio: bool
    duration_seconds: float | None


def probe_media(path: Path) -> MediaInfo:
    """Read stream metadata, falling back to container duration, and close the file.

    A missing duration remains unknown; decoding or estimating it belongs to the
    caller. Invalid containers propagate their PyAV error.
    """
    with av.open(str(path)) as container:
        durations = [
            float(stream.duration * stream.time_base)
            for stream in (*container.streams.video, *container.streams.audio)
            if stream.duration is not None and stream.time_base is not None
        ]
        duration = max(durations) if durations else None
        if duration is None and container.duration is not None:
            duration = float(container.duration / av.time_base)
        return MediaInfo(bool(container.streams.video), bool(container.streams.audio), duration)


def video_size(path: Path) -> tuple[int, int]:
    """Read the first video stream's pixel (width,height) without decoding frames."""
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Spatial bucketing requires a video stream")
        stream = container.streams.video[0]
        width, height = stream.codec_context.width, stream.codec_context.height
        if min(width, height) < 1:
            raise ValueError("Video stream has no positive declared spatial geometry")
        return width, height
