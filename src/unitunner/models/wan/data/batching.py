"""Group native Wan temporal/spatial grids into resumable shuffled epochs."""

from collections import defaultdict
from pathlib import Path

from unitunner.artifacts.tensors import TensorReader
from unitunner.data.cache.storage import CacheEntry
from unitunner.data.datasets.batching import EpochBatches
from unitunner.models.wan.data.cache.storage import WanCacheManifest


def build_wan_batches(
    directory: Path,
    manifest: WanCacheManifest,
    *,
    batch_size: int,
    repeats: int,
    seed: int = 42,
) -> EpochBatches[CacheEntry]:
    """Read verified CFHW headers, repeat each grid and retain shuffled short tails.

    The caller preflights complete records first. Captions remain variable length;
    target temporal/spatial geometry determines the native stackable batch key.
    No payload copies, model RNG draws or mutable sampler checkpoint are needed.
    """
    if repeats < 1:
        raise ValueError("Wan repeats must be positive")
    groups: dict[tuple[int, ...], list[CacheEntry]] = defaultdict(list)
    for entry in manifest.entries:
        with TensorReader(directory / entry.file) as reader:
            shape = reader.shape("latents")
        groups[shape].extend([entry] * repeats)
    return EpochBatches(groups, batch_size=batch_size, seed=seed)
