"""Plan shuffled single-record H3 epochs without merging variable AV layouts."""

from unituner.data.cache.storage import CacheEntry
from unituner.data.datasets.batching import EpochBatches
from unituner.models.minimax_h3.data.cache.storage import MiniMaxH3CacheManifest


def build_minimax_h3_batches(
    manifest: MiniMaxH3CacheManifest,
    *,
    batch_size: int,
    repeats: int,
    seed: int = 42,
) -> EpochBatches[CacheEntry]:
    """Shuffle repeat-expanded records under the native batch-one training contract.

    Each record is a complete variable-length AV sample; accumulation belongs to
    the shared engine. The microbatch cursor reconstructs ordering on resume.
    """
    if batch_size != 1 or repeats < 1:
        raise ValueError("MiniMaxH3 training requires batch_size=1 and positive repeats")
    return EpochBatches({(0,): manifest.entries * repeats}, batch_size=1, seed=seed)
