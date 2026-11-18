"""Read native Wan inference control frames with explicit length and pixel semantics."""

from pathlib import Path

import torch
from torch import Tensor

from unituner.config.datasets import VideoDatasetConfig
from unituner.data.media.video import read_video


def read_wan_control(path: Path, resolution: tuple[int, int], *, frames: int) -> Tensor:
    """Return CPU FP32 RGB CTHW control pixels using native inference normalization.

    Decode the first requested frames without FPS conversion, resize/crop with
    the shared native bucket policy and reject short sources. Normalize each
    uint8 value by255, subtract0.5 and divide0.5 in that order; the cache's
    /127.5-1 expression has different FP32 rounding.
    """
    raw = read_video(path, resolution, VideoDatasetConfig(), end_frame=frames).pixels
    if len(raw) != frames:
        raise ValueError(f"Wan control_video requires at least {frames} frames; decoded {len(raw)}")
    return torch.from_numpy(raw).permute(3, 0, 1, 2).contiguous().float().div_(255).sub_(0.5).div_(0.5)
