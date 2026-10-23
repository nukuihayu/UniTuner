"""Atomically encode native RGB video pixels with explicit frame timing and container choices."""

from fractions import Fraction
from pathlib import Path

import av
import torch
from torch import Tensor

from unitunner.artifacts.atomic import atomic_output


def save_video(pixels: Tensor, path: Path, *, fps: int) -> None:
    """Write finite floating CTHW RGB[-1,1] pixels as MP4/H264 or lossless MKV/FFV1.

    Stage one frame at a time, clamp then round to nearest uint8 RGB, with explicit integer-frame PTS at the
    requested positive integer FPS. MP4 uses yuv420p and is lossy; MKV/bgr0 keeps
    the quantized RGB values. Flush the encoder before atomically replacing the
    destination; failures leave its old contents intact and close all resources.
    """
    if (
        pixels.ndim != 4
        or pixels.shape[0] != 3
        or min(pixels.shape) < 1
        or not pixels.is_floating_point()
        or not torch.isfinite(pixels).all()
        or type(fps) is not int
        or fps < 1
    ):
        raise ValueError("Video output requires finite floating RGB CTHW pixels and positive integer fps")
    suffix = path.suffix.lower()
    if suffix not in (".mp4", ".mkv") or (suffix == ".mp4" and any(size % 2 for size in pixels.shape[-2:])):
        raise ValueError("Video output requires .mp4 with even dimensions or .mkv")
    with (
        atomic_output(path) as temporary,
        av.open(str(temporary), mode="w", format="mp4" if suffix == ".mp4" else "matroska") as container,
    ):
        stream = container.add_stream("libx264" if suffix == ".mp4" else "ffv1", rate=fps)
        stream.width, stream.height = pixels.shape[-1], pixels.shape[-2]
        stream.pix_fmt = "yuv420p" if suffix == ".mp4" else "bgr0"
        for index, values in enumerate(pixels.detach().unbind(dim=1)):
            array = ((values.float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
