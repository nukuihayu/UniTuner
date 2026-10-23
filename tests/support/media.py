"""Create deterministic audio/video fixtures for media and training tests."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def write_media(
    path: Path,
    *,
    fps: int = 24,
    video_ticks: tuple[int, ...] = (),
    audio_rate: int | None = None,
    audio_channels: int = 2,
    audio_start: int = 0,
    audio_samples: int = 0,
    audio_jump: int = 0,
) -> None:
    """Write deterministic lossless RGB and PCM streams with explicit independent PTS clocks."""
    with av.open(str(path), "w") as container:
        # Declare every stream before muxing writes the container header.
        video_stream = None
        if video_ticks:
            video_stream = container.add_stream("ffv1", rate=fps)
            video_stream.width, video_stream.height, video_stream.pix_fmt = 80, 48, "bgr0"
        audio_stream = None
        if audio_rate is not None:
            audio_stream = container.add_stream("pcm_s16le", rate=audio_rate)
            audio_stream.layout = "mono" if audio_channels == 1 else "stereo"
        if video_stream is not None:
            stream = video_stream
            for index, tick in enumerate(video_ticks):
                pixels = (np.arange(48 * 80 * 3).reshape(48, 80, 3) * 3 + index * 17).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts, frame.time_base = tick, Fraction(1, fps)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        if audio_stream is not None:
            stream = audio_stream
            samples = np.arange(audio_samples, dtype=np.float64)
            waveform = np.stack([np.sin(samples * (0.013 + i * 0.004)) * 16000 for i in range(audio_channels)]).astype(
                np.int16
            )
            for start in range(0, audio_samples, 1024):
                frame = av.AudioFrame.from_ndarray(
                    waveform[:, start : start + 1024].copy(), format="s16p", layout=stream.layout.name
                )
                frame.sample_rate = audio_rate  # pyright: ignore[reportAttributeAccessIssue] -- PyAV exposes a writable descriptor, but its stub marks it read-only.
                frame.time_base = Fraction(1, audio_rate)
                frame.pts = audio_start + start + (audio_jump if start >= 5120 else 0)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
