"""Shared caption caching and image publication; families own conditioning and decoding."""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from unituner.data.media.image import save_image


class CaptionCache[Captions]:
    """Commit CPU caption features only after a complete successful encoding."""

    captions: Captions | None = None

    def cached_captions(self, load: Callable[[], Captions]) -> Captions:
        if self.captions is None:
            with torch.autocast("cpu", enabled=False):
                self.captions = load()
        return self.captions


def publish_images[Sample](
    samples: Iterable[Sample],
    render: Callable[[Sample], Iterable[Tensor]],
    *,
    directory: Path,
    update: int,
    rounding: Literal["nearest", "truncate"],
    save: Callable[..., None] = save_image,
) -> None:
    """Render in sample order and save every decoded layer using stable filenames."""
    for index, sample in enumerate(samples):
        for layer, pixels in enumerate(render(sample)):
            suffix = "" if layer == 0 else f".layer-{layer:03d}"
            save(pixels, directory / f"step-{update:08d}-{index:02d}{suffix}.png", rounding=rounding)
