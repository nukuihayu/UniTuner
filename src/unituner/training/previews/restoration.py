"""Index restoration preview samples once and render deterministic update pages."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from unituner.config.training import TrainingConfig
from unituner.data.datasets.restoration import AlignedImagePair, RestorationDataset
from unituner.data.media.image import save_image
from unituner.logging import operation_context
from unituner.progress import status

if TYPE_CHECKING:
    from unituner.training.methods.restoration import RestorationMethod


@dataclass(frozen=True, slots=True)
class PreviewSample:
    """Pin a source and its original read options, independent of dataset repeats."""

    pair: AlignedImagePair
    seed: int
    resolution: tuple[int, int]
    horizontal_flip_probability: float


def _comparison(low: Tensor, restored: Tensor, high: Tensor) -> Tensor:
    """Build a CPU RGB LQ/restored/HQ strip with two-pixel black grid borders."""
    if low.shape != restored.shape or low.shape != high.shape or low.ndim != 3 or (low.shape[0] != 3):
        raise ValueError("Restoration preview requires matching RGB CHW images")
    _, height, width = low.shape
    grid = torch.full((3, height + 4, 3 * width + 8), -1.0, dtype=torch.float32)
    for index, pixels in enumerate((low, restored, high)):
        left = 2 + index * (width + 2)
        grid[:, 2 : height + 2, left : left + width] = pixels.detach().float().cpu()
    return grid


class RestorationPreviews:
    """Own a reusable CPU index; the caller owns inference precision and RNG isolation."""

    def __init__(self, config: TrainingConfig, dataset: RestorationDataset) -> None:
        preview = config.preview
        if preview is None:
            raise ValueError("Restoration previews require preview configuration")
        groups = {Path(group.cache_directory): group for group in config.dataset.groups.values()}
        unique = {}
        for position, pair in enumerate(dataset.pairs):
            unique.setdefault(pair, position)
        candidates = list(unique.items())
        if not candidates:
            raise ValueError("Restoration previews require nonempty dataset items")
        random.Random(config.train.seed).shuffle(candidates)
        samples = []
        for pair, position in candidates:
            if pair.cache_directory not in groups:
                raise ValueError(f"No restoration dataset group for cache {pair.cache_directory}")
            group = groups[pair.cache_directory]
            samples.append(
                PreviewSample(pair, config.train.seed + position, group.resolution, group.horizontal_flip_probability)
            )
        self.samples = tuple(samples)
        self.dataset = dataset
        self.preview = preview
        self.seed = config.train.seed
        self.directory = Path(config.output.directory) / "previews"

    def select(self, update: int) -> tuple[tuple[int, PreviewSample], ...]:
        """Derive pages from update alone, including final off-cadence previews."""
        if type(update) is not int or update < 0:
            raise ValueError("Preview update must be a nonnegative integer")
        count = min(self.preview.num_images, len(self.samples))
        round_index = max(
            0, (update + self.preview.every_updates - 1) // self.preview.every_updates - int(not self.preview.at_start)
        )
        indices = ((round_index * count + index) % len(self.samples) for index in range(count))
        return tuple((index, self.samples[index]) for index in indices)

    def __call__(self, update: int, method: RestorationMethod, device: torch.device) -> None:
        """Borrow the active restoration method to publish LQ/restored/HQ strips."""
        for index, (selected, sample) in enumerate(self.select(update)):
            pair = sample.pair
            status(
                f"[preview] update={update} sample={selected + 1}/{len(self.samples)} "
                f"source={pair.high} seed={self.seed}"
            )
            with operation_context(update=update, source=pair.high, phase="preview"):
                low, high = self.dataset.read(pair, sample.resolution, sample.seed, sample.horizontal_flip_probability)
                caption = method.captions((pair,))
                latent = method.encode(low.unsqueeze(0).to(device))
                restored = method.decode(method.restore(latent, caption))[0]
                save_image(_comparison(low, restored, high), self.directory / f"step-{update:08d}-{index:02d}.png")
