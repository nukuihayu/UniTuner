# Copyright 2025-2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synchronize decoded native H3 audio/video and atomically publish timed AV files."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import torch
from PIL import Image
from torch import Tensor

from unitunner.artifacts.atomic import atomic_output


@dataclass(frozen=True, slots=True)
class MiniMaxH3DecodedImage:
    """Own one CPU uint8 HWC RGB image, without an unused audio waveform."""

    image: Tensor


@dataclass(frozen=True, slots=True)
class MiniMaxH3DecodedAV:
    """Own CPU uint8 FHWC RGB frames and contiguous FP32 stereo2L samples."""

    video: Tensor
    audio: Tensor
    fps: int = 24
    sample_rate: int = 32000


def save_minimax_h3_media(decoded: MiniMaxH3DecodedImage | MiniMaxH3DecodedAV, path: Path) -> None:
    """Atomically publish one native PNG image or H264/AAC MP4/MKV result.

    Borrow CPU uint8 RGB image storage without normalization or a second
    rounding step. Reject malformed image geometry/dtype before creating any
    output; all encoders and temporary files close on every exit.
    """
    if isinstance(decoded, MiniMaxH3DecodedAV):
        save_minimax_h3_av(decoded, path)
        return
    pixels = decoded.image
    if (
        pixels.device.type != "cpu"
        or pixels.dtype != torch.uint8
        or pixels.ndim != 3
        or pixels.shape[-1] != 3
        or min(pixels.shape) <= 0
        or path.suffix.lower() != ".png"
    ):
        raise ValueError("MiniMaxH3 image output requires CPU uint8 HWC RGB and .png")
    with atomic_output(path) as temporary, Image.fromarray(pixels.numpy()) as image:
        image.save(temporary, format="PNG")


def synchronize_decoded_av(
    video: Tensor, audio: Tensor, *, frame_count: int, fps: int = 24, sample_rate: int = 32000
) -> MiniMaxH3DecodedAV:
    """Trim native B3FHW pixels/B2L waveforms to their shared decoded extent.

    Use exact Fraction duration and Python's nearest-even rounding in original
    order. Quantize video16frames at a time: clamp[-1,1], add1, multiply127.5,
    round, uint8, FHWC. Trim audio by the minimum planned/decoded extent, then
    clamp[-1,1]. Return independent contiguous CPU storage without input mutation.
    """
    if any(type(value) is not int or value <= 0 for value in (frame_count, fps, sample_rate)):
        raise ValueError("MiniMaxH3 decoded output requires positive integer frame count/fps/sample rate")
    if (
        video.ndim != 5
        or video.shape[:2] != (1, 3)
        or min(video.shape) < 1
        or audio.ndim != 3
        or audio.shape[:2] != (1, 2)
        or min(audio.shape) < 1
        or not video.is_floating_point()
        or not audio.is_floating_point()
        or not torch.isfinite(video).all()
        or not torch.isfinite(audio).all()
    ):
        raise ValueError("MiniMaxH3 decoded output requires finite B3FHW pixels and B2L stereo waveforms with B=1")
    planned_samples = round(Fraction(frame_count * sample_rate, fps))
    decoded_samples = round(Fraction(video.shape[2] * sample_rate, fps))
    samples = min(audio.shape[-1], planned_samples, decoded_samples)
    if samples <= 0:
        raise ValueError("MiniMaxH3 decoded audio/video duration is empty")
    frames = min(frame_count, video.shape[2], max(1, round(Fraction(samples * fps, sample_rate))))
    chunks = []
    for chunk in video[0, :, :frames].detach().cpu().split(16, dim=1):
        chunk = chunk.float().clamp(-1.0, 1.0)
        chunks.append(((chunk + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0))
    pixels = torch.cat(chunks).contiguous()
    waveform = audio[0, :, :samples].detach().cpu().float().clamp(-1.0, 1.0).contiguous()
    return MiniMaxH3DecodedAV(pixels, waveform, fps, sample_rate)


def save_minimax_h3_av(decoded: MiniMaxH3DecodedAV, path: Path) -> None:
    """Atomically mux H264/CRF16/yuv420p video and AAC stereo audio with explicit PTS.

    Preserve native lossy VAE choices for MP4/MKV. Flush both encoders before
    publication; a failure closes the container and preserves an old destination.
    Borrow finite CPU samples and uint8 RGB frames without mutation or RNG draws.
    """
    video, audio = decoded.video, decoded.audio
    if (
        video.device.type != "cpu"
        or video.dtype != torch.uint8
        or video.ndim != 4
        or video.shape[-1] != 3
        or min(video.shape) < 1
        or any(size % 2 for size in video.shape[1:3])
        or audio.device.type != "cpu"
        or audio.dtype != torch.float32
        or audio.ndim != 2
        or audio.shape[0] != 2
        or audio.shape[1] < 1
        or not torch.isfinite(audio).all()
        or type(decoded.fps) is not int
        or decoded.fps <= 0
        or type(decoded.sample_rate) is not int
        or decoded.sample_rate <= 0
        or path.suffix.lower() not in (".mp4", ".mkv")
    ):
        raise ValueError(
            "MiniMaxH3 AV output requires CPU uint8 RGB with even dimensions, finite FP32 stereo and .mp4/.mkv"
        )
    with (
        atomic_output(path) as temporary,
        av.open(str(temporary), mode="w", format="mp4" if path.suffix.lower() == ".mp4" else "matroska") as container,
    ):
        video_stream = container.add_stream("libx264", rate=decoded.fps, options={"crf": "16"})
        video_stream.width, video_stream.height = video.shape[2], video.shape[1]
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=decoded.sample_rate)
        audio_stream.layout = "stereo"
        for index, pixels in enumerate(video):
            frame = av.VideoFrame.from_ndarray(pixels.numpy(), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, decoded.fps)
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)
        for start in range(0, audio.shape[1], 1024):
            samples = audio[:, start : start + 1024].contiguous().numpy()
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
            frame.sample_rate = decoded.sample_rate
            frame.pts, frame.time_base = start, Fraction(1, decoded.sample_rate)
            for packet in audio_stream.encode(frame):
                container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)
