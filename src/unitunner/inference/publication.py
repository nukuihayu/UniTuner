"""Publish decoded image layers without model-specific loading or sampling."""

import json
from pathlib import Path
from typing import Literal

from torch import Tensor

from unitunner.data.media.image import save_image


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
    print(
        json.dumps({"generated": True, "width": pixels.shape[-1], "height": pixels.shape[-2], "images": destinations})
    )
    return 0
