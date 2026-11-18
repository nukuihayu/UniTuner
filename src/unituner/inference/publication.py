"""Publish decoded image layers without model-specific loading or sampling."""

from pathlib import Path
from typing import Literal

from torch import Tensor

from unituner.data.media.image import save_image
from unituner.logging import publish_result


def publish_images(pixels: Tensor, output: Path, *, rounding: Literal["nearest", "truncate"]) -> int:
    """Write BCHW RGB/RGBA layers and report the actual saved paths.

    Pixels are decoded floats in [-1, 1]. The caller supplies the native uint8
    rounding rule; layer zero uses output and later layers use .layer-NNN suffixes.
    """
    destinations = []
    for index, frame in enumerate(pixels):
        destination = output if index == 0 else output.with_name(f"{output.stem}.layer-{index:03d}{output.suffix}")
        save_image(frame, destination, rounding=rounding)
        destinations.append(str(destination))
    publish_result(
        {"generated": True, "width": pixels.shape[-1], "height": pixels.shape[-2], "images": destinations},
        stage="generate",
        message=f"Saved {len(destinations)} image(s), {pixels.shape[-1]}x{pixels.shape[-2]}: {output}",
    )
    return 0
