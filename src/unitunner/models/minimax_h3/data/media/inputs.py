# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare native H3 VAE pixels and reference canvases without mutating decoded media."""

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

from unitunner.data.media.image import resize_pixels
from unitunner.data.media.video import read_timestamp_video
from unitunner.models.minimax_h3.layout import TARGET_FPS

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344


def read_condition_image(path: Path, *, resolution: tuple[int, int]) -> Tensor:
    """Read owned CPU uint81HWC RGB using native cover resize and center crop.

    FL generation shares training bucket preprocessing. Convert to RGB before
    resize; preserve native Lanczos and half-up dimension rounding.
    No EXIF transpose, alpha compositing or posterior target encoding occurs.
    """
    _validate_size(resolution)
    with Image.open(path) as image, image.convert("RGB") as rgb:
        return torch.from_numpy(resize_pixels(np.array(rgb), resolution)).unsqueeze(0)


def round_to_canvas_multiple(value: float, *, multiple: int = CANVAS_MULTIPLE) -> int:
    """Round a positive finite extent to the nearest grid point, at least one cell.

    Preserve Python's ties-to-even rule used by the released H3 implementation.
    """
    if not math.isfinite(value) or value <= 0 or type(multiple) is not int or multiple <= 0:
        raise ValueError("MiniMaxH3 canvas rounding requires a positive finite extent and positive integer multiple")
    return max(multiple, round(value / multiple) * multiple)


def _validate_size(size: tuple[int, int]) -> None:
    if len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
        raise ValueError("MiniMaxH3 canvas requires positive integer width/height")


def adapt_reference_canvas(width: int, height: int) -> tuple[int, int]:
    """Return native video-reference width/height on the32-pixel grid.

    Start at768 pixels on the short edge, apply the768x1344 nominal area
    budget, then round each side independently. Preserve the released rounding
    order; the rounded area may slightly exceed the nominal continuous budget.
    """
    _validate_size((width, height))
    ratio = width / height
    if ratio >= 1.0:
        nominal_width, nominal_height = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nominal_width, nominal_height = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nominal_width * nominal_height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    return round_to_canvas_multiple(nominal_width), round_to_canvas_multiple(nominal_height)


def resize_frames(frames: Sequence[NDArray[np.uint8]], size: tuple[int, int]) -> Tensor:
    """Lanczos-resize borrowed uint8 HWC RGB/RGBA frames to owned CPU THWC RGB.

    Drop alpha before resizing, without compositing, cropping or implicit EXIF
    rotation. Every frame/image closes on error; the output owns contiguous
    storage and no input array is altered or retained by the output.
    """
    _validate_size(size)
    if not frames:
        raise ValueError("MiniMaxH3 resize requires at least one frame")
    resized = []
    for frame in frames:
        if frame.dtype != np.uint8 or frame.ndim != 3 or min(frame.shape) <= 0 or frame.shape[-1] not in (3, 4):
            raise ValueError("MiniMaxH3 resize requires nonempty uint8 HWC RGB/RGBA frames")
        with Image.fromarray(frame[..., :3]) as image, image.resize(size, Image.Resampling.LANCZOS) as fitted:
            resized.append(torch.from_numpy(np.array(fitted)))
    return torch.stack(resized)


def read_reference_image(path: Path, *, target_size: tuple[int, int]) -> Tensor:
    """Read a native H3 image reference and fit its area to the target canvas.

    Preserve source aspect ratio and the downscale-only area factor before
    nearest32-grid rounding. Tiny axes can therefore round up to32 pixels.
    Return owned CPU uint8 THWC RGB pixels without consuming RNG.
    """
    _validate_size(target_size)
    with Image.open(path) as image, image.convert("RGB") as rgb:
        width, height = rgb.size
        scale = min(1.0, math.sqrt((target_size[0] * target_size[1]) / (width * height)))
        size = round_to_canvas_multiple(width * scale), round_to_canvas_multiple(height * scale)
        return resize_frames([np.asarray(rgb)], size)


def read_reference_video(path: Path, *, target_frame_count: int) -> Tensor:
    """Read a native24Hz H3 video reference, trim to17*n+5 and resize its canvas.

    The caller supplies the reference frame cap explicitly, including the360-frame
    cap for a one-frame target. Decode timestamp sampling precedes trimming and
    Lanczos resize. Preserve the native downscale-only source-area check and
    nearest32-grid rounding. Return owned CPU uint8 THWC RGB pixels.
    """
    if type(target_frame_count) is not int or target_frame_count < 5:
        raise ValueError("MiniMaxH3 reference video cap requires at least5 frames")
    decoded = read_timestamp_video(path, target_fps=float(TARGET_FPS))
    usable = min(len(decoded.frames), target_frame_count)
    if usable < 5:
        raise ValueError("MiniMaxH3 reference video requires at least5 frames")
    usable = 5 + ((usable - 5) // 17) * 17
    frames = decoded.frames[:usable]
    source_height, source_width = frames[0].shape[:2]
    width, height = adapt_reference_canvas(source_width, source_height)
    if source_width * source_height < width * height:
        width, height = round_to_canvas_multiple(source_width), round_to_canvas_multiple(source_height)
    return resize_frames(frames, (width, height))


def prepare_pixels(frames: Tensor | NDArray[np.uint8] | NDArray[np.floating]) -> Tensor:
    """Convert borrowed THWC RGB/RGBA to owned contiguous FP32 B3THW in[-1,1].

    Uint8 follows FP32 /127.5 then -1. Floating[0,1] follows cast-to-FP32,
    *2 then -1; preserve that order for FP16/FP64 inputs. Drop alpha without
    compositing. Output retains the Tensor input's device and gradients through
    floating RGB channels; alpha gradients are zero. Caller storage, including
    FP32/noncontiguous/read-only NumPy views, is never modified.
    """
    if isinstance(frames, np.ndarray) and not frames.flags.writeable:
        frames = frames.copy()
    pixels = torch.as_tensor(frames)
    if pixels.ndim != 4 or min(pixels.shape) <= 0 or pixels.shape[-1] not in (3, 4):
        raise ValueError("MiniMaxH3 decoded pixels require nonempty THWC RGB/RGBA")
    pixels = pixels[..., :3]
    if pixels.dtype == torch.uint8:
        pixels = pixels.float().div_(127.5).sub_(1.0)
    elif pixels.is_floating_point():
        if not torch.all((pixels >= 0) & (pixels <= 1)):
            raise ValueError("MiniMaxH3 floating decoded pixels must be finite and in[0,1]")
        pixels = pixels.float().mul(2.0).sub_(1.0)
    else:
        raise ValueError("MiniMaxH3 decoded pixels require uint8 or floating dtype")
    return pixels.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
