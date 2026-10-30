"""Discover images and preserve native resize/crop and RGB normalization semantics."""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

from unitunner.artifacts.atomic import atomic_output

_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"})


def discover_images(directory: Path, *, caption_extension: str | None = None) -> tuple[Path, ...]:
    """Return sorted immediate image files; no recursion, decoder probes or global plugins."""
    files = tuple(sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in _EXTENSIONS))
    if caption_extension is not None:
        files = tuple(path for path in files if path.with_suffix(caption_extension).is_file())
    if not files:
        raise ValueError("Image source directory contains no supported images with the requested caption policy")
    return files


def index_conditions(targets: tuple[Path, ...], directory: Path) -> dict[str, Path]:
    """Match complete nonempty unique file stems, rejecting missing/extra images."""
    files = discover_images(directory)
    target_names = {path.stem for path in targets}
    indexed = {path.stem: path for path in files}
    if len(target_names) != len(targets) or len(indexed) != len(files) or indexed.keys() != target_names:
        raise ValueError("Condition images require identical complete unique target stems")
    return indexed


def resize_pixels(
    pixels: NDArray[np.uint8], resolution: tuple[int, int], *, resample: Literal["image", "mask"] = "image"
) -> NDArray[np.uint8]:
    """Return owned uint8 HWC RGB(A) using native Lanczos-up/area-down bucket geometry.

    Round resized extents half up and center-crop. Preserve alpha and channel
    order without compositing, EXIF handling or mutation of caller storage.
    Mask mode shares all geometry but uses nearest interpolation for labels.
    """
    if pixels.ndim != 3 or pixels.shape[-1] not in (3, 4) or pixels.dtype != np.uint8 or min(pixels.shape) < 1:
        raise ValueError("Resize pixels requires nonempty uint8 HWC RGB/RGBA")
    if min(resolution) < 1:
        raise ValueError("Image resolution dimensions must be positive")
    if resample not in ("image", "mask"):
        raise ValueError("Resample mode must be image or mask")
    bucket_width, bucket_height = resolution
    height, width = pixels.shape[:2]
    if (width, height) == resolution:
        return pixels.copy()
    scale = max(bucket_width / width, bucket_height / height)
    width, height = int(width * scale + 0.5), int(height * scale + 0.5)
    if scale > 1 or resample == "mask":
        interpolation = Image.Resampling.NEAREST if resample == "mask" else Image.Resampling.LANCZOS
        with Image.fromarray(pixels) as image, image.resize((width, height), interpolation) as resized:
            values = np.array(resized)
    else:
        values = cv2.resize(pixels, (width, height), interpolation=cv2.INTER_AREA)
    left, top = (width - bucket_width) // 2, (height - bucket_height) // 2
    return np.ascontiguousarray(values[top : top + bucket_height, left : left + bucket_width], dtype=np.uint8)


def read_image(
    path: Path, resolution: tuple[int, int], *, rgb_before_resize: bool = False, channels: Literal[3, 4] = 3
) -> Tensor:
    """Decode to CPU FP32 CHW [-1,1] at a (width,height) bucket.

    Preserve historical orientation and alpha-removal order. Upscale with
    Pillow Lanczos, downscale with OpenCV area, round resized dimensions half
    up, then center crop. No EXIF rotation or alpha compositing is implicit.
    Return independent contiguous storage and close every decoded image.
    Set channels=4 to append opaque alpha before resize for RGB inputs and retain
    source alpha for RGBA inputs. Set rgb_before_resize for providers that discard alpha
    before resampling; the default retains historical image-cache ordering.
    """
    if min(resolution) <= 0:
        raise ValueError("Image resolution dimensions must be positive")
    if channels not in (3, 4) or (channels == 4 and rgb_before_resize):
        raise ValueError("Image channels must be 3 or 4; RGB-before-resize cannot retain alpha")
    with Image.open(path) as decoded:
        source = (
            decoded.convert("RGBA")
            if channels == 4
            else decoded.copy()
            if decoded.mode in {"RGB", "RGBA"} and not rgb_before_resize
            else decoded.convert("RGB")
        )
        try:
            pixels = resize_pixels(np.array(source), resolution)
            return (
                torch.from_numpy(np.ascontiguousarray(pixels[..., :channels])).permute(2, 0, 1).contiguous().float()
                / 127.5
                - 1
            )
        finally:
            source.close()


def save_image(pixels: Tensor, path: Path, *, rounding: Literal["nearest", "truncate"] = "nearest") -> None:
    """Atomically encode finite CHW RGB/RGBA [-1,1] pixels to PNG, clamping decoder overshoot.

    Quantize to nearest uint8 by default, or truncate for native FLUX3 output,
    after mapping [-1,1] to [0,255]. The source tensor,
    its device, gradients and the previous destination remain intact on failure.
    """
    if (
        pixels.ndim != 3
        or pixels.shape[0] not in (3, 4)
        or not pixels.is_floating_point()
        or not torch.isfinite(pixels).all()
    ):
        raise ValueError("PNG output requires finite floating RGB/RGBA CHW pixels")
    if rounding not in ("nearest", "truncate"):
        raise ValueError("PNG rounding must be nearest or truncate")
    values = (pixels.detach().float().cpu().clamp(-1, 1) + 1) * 127.5
    if rounding == "nearest":
        values = values.round()
    array = values.to(torch.uint8).permute(1, 2, 0).numpy()
    with Image.fromarray(array) as image, atomic_output(path) as temporary:
        image.save(temporary, format="PNG")
