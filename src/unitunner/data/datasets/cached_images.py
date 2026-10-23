"""Index multiple immutable image caches into shape-safe shuffled training epochs."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from unitunner.artifacts.hashing import canonical_sha256, file_sha256
from unitunner.artifacts.tensors import TensorReader
from unitunner.config.datasets import DataConfig
from unitunner.data.cache.storage import CacheEntry, load_latent_manifest


@dataclass(frozen=True, slots=True)
class CachedImage:
    """Keep a record attached to its source directory across shuffles and repeats."""

    directory: Path
    entry: CacheEntry


def index_cached_images(
    dataset: DataConfig,
    *,
    family: str,
    variant: str,
    version: str | None,
) -> tuple[dict[tuple[int, ...], list[CachedImage]], dict[tuple[int, ...], int], str]:
    """Validate family/variant/version and group each dataset by complete tensor geometry.

    Captions remain variable length. Conditions, references and layered targets
    participate in the key so native batch consumers never stack unequal grids.
    Repeats occur before batching; independent datasets retain independent tails.
    Return groups and a content digest covering every ordered source manifest.
    """
    groups: dict[tuple[int, ...], list[CachedImage]] = defaultdict(list)
    digests: list[JsonValue] = []
    batch_sizes: dict[tuple[int, ...], int] = {}
    for source_index, (name, source) in enumerate(dataset.groups.items()):
        if source.cache_directory is None:
            raise ValueError("Image source requires a cache_directory")
        directory = Path(source.cache_directory)
        manifest = load_latent_manifest(directory)
        if manifest.family != family or manifest.version != version:
            raise ValueError("Training cache family/version differs from the selected model")
        if family == "moss-image" and manifest.variant != ("edit" if variant == "edit" else None):
            raise ValueError("MOSS cache edit contract differs from the training variant")
        if family in ("flux", "flux2", "flux3", "qwen-image") and manifest.variant != variant:
            raise ValueError("Training cache variant differs from the selected model")
        digests.append({"group": name, "manifest": file_sha256(directory / "manifest.json")})
        for entry in manifest.entries:
            with TensorReader(directory / entry.file) as reader:
                shapes = tuple(reader.shape(key) for key in reader.keys() if key != "caption")
            key = (source_index, len(shapes), *(dimension for shape in shapes for dimension in (len(shape), *shape)))
            batch_sizes[key] = source.batch_size
            groups[key].extend([CachedImage(directory, entry)] * source.repeats)
    return dict(groups), batch_sizes, canonical_sha256(digests)
