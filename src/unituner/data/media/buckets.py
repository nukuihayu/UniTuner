"""Select model-aligned image buckets using area budgets and aspect ratios."""

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True, slots=True)
class ImageBuckets:
    """Match Musubi area/aspect buckets; dimensions are pixel (width, height).

    Small no-upscale images must contain at least one alignment unit on each
    axis. Disabled buckets always resize/crop to the configured resolution.
    """

    resolution: tuple[int, int]
    alignment: int = 16
    enabled: bool = True
    no_upscale: bool = False
    caption_extension: str | None = None
    prompt: str | None = None

    @property
    def required_caption_extension(self) -> str | None:
        """A group prompt removes the need for any per-file caption."""
        return None if self.prompt is not None else self.caption_extension

    def caption(self, path: Path) -> str:
        """Resolve the effective caption, preserving an explicitly empty group prompt."""
        if self.prompt is not None:
            return self.prompt
        caption = path.with_suffix(self.caption_extension or ".txt")
        return caption.read_text(encoding="utf-8") if caption.is_file() else ""

    def select(self, size: tuple[int, int]) -> tuple[int, int]:
        """Return the nearest sorted aspect bucket, or aligned source-sized bucket."""
        if min(*size, *self.resolution, self.alignment) < 1:
            raise ValueError("Image and bucket dimensions must be positive")
        if any(value % self.alignment for value in self.resolution):
            raise ValueError("Bucket resolution must be divisible by model alignment")
        if not self.enabled:
            return self.resolution
        area = math.prod(self.resolution)
        if self.no_upscale and math.prod(size) <= area:
            result = tuple(value // self.alignment * self.alignment for value in size)
            if min(result) == 0:
                raise ValueError("No-upscale image is smaller than model alignment")
            return result[0], result[1]
        extent = math.isqrt(area)
        minimum = max(self.alignment, extent // 2 // self.alignment * self.alignment)
        buckets: set[tuple[int, int]] = set()
        for width in range(minimum, extent + self.alignment, self.alignment):
            height = max(self.alignment, area // width // self.alignment * self.alignment)
            buckets.update(((width, height), (height, width)))
        return min(sorted(buckets), key=lambda bucket: abs(bucket[0] / bucket[1] - size[0] / size[1]))

    def for_image(self, path: Path) -> tuple[int, int]:
        """Read only the image header and resolve its target bucket."""
        with Image.open(path) as image:
            return self.select(image.size)
