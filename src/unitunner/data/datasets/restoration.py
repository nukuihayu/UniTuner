"""Index aligned low-quality/high-quality image pairs and pin source-to-caption content associations."""

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue
from torch import Tensor

from unitunner.artifacts.hashing import canonical_sequence_sha256, canonical_sha256, file_sha256
from unitunner.data.cache.storage import CacheEntry, read_latent_manifest, validate_latent_entry
from unitunner.data.degradation.config import DegradationConfig
from unitunner.progress import progress


@dataclass(frozen=True, slots=True)
class AlignedImagePair:
    """Pin an HQ source, its caption record and either aligned LQ or synthetic degradation."""

    low: Path | None
    high: Path
    caption: CacheEntry
    cache_directory: Path
    degradation: DegradationConfig | None = None


@dataclass(frozen=True, slots=True)
class RestorationDataset:
    """Own verified pair indices and a decoder supplied at the optional media boundary."""

    pairs: tuple[AlignedImagePair, ...]
    digest: str
    read: Callable[[AlignedImagePair, tuple[int, int], int, float], tuple[Tensor, Tensor]]


def index_restoration_pairs(
    high_files: tuple[Path, ...],
    low_files: tuple[Path, ...] | None,
    cache: Path,
    *,
    degradation: DegradationConfig | None = None,
    family: str | None = None,
    caption_extension: str | None = None,
    prompt: str | None = None,
) -> tuple[tuple[AlignedImagePair, ...], str]:
    """Match unique stems or synthetic HQ sources, pin captions and hash numerical inputs.

    Inputs are explicitly ordered discovered image files. A cache produced from a
    different source/caption list is rejected before training; basename coincidence
    never suffices. Low and high file bytes both participate in the returned digest.
    """
    high = {path.stem for path in high_files}
    synthetic = degradation is not None and degradation.enabled
    if synthetic != (low_files is None):
        raise ValueError("Image restoration requires either low-quality files or enabled synthetic degradation")
    low = {} if low_files is None else {path.stem: path for path in low_files}
    if (
        not high
        or len(high) != len(high_files)
        or (low_files is not None and (len(low) != len(low_files) or high != low.keys()))
    ):
        raise ValueError("Aligned images require identical nonempty unique image stems")
    del high
    manifest = read_latent_manifest(cache)
    if family is not None and manifest.family != family:
        raise ValueError("Restoration caption cache family differs from the selected model")
    if len(manifest.entries) != len(high_files):
        raise ValueError("Restoration source count differs from caption cache manifest")
    pairs = []

    def records() -> Iterator[JsonValue]:
        for path, entry in zip(
            progress(high_files, "[train] Validating restoration pairs"), manifest.entries, strict=True
        ):
            caption_path = path.with_suffix(caption_extension or ".txt")
            caption = (
                prompt
                if prompt is not None
                else (caption_path.read_text(encoding="utf-8") if caption_path.is_file() else "")
            )
            content: dict[str, JsonValue] = {
                "image": file_sha256(path, release_cache=True),
                "caption_utf8": hashlib.sha256(caption.encode()).hexdigest(),
            }
            if manifest.family in ("flux2", "flux3"):
                content["references"] = []
            if manifest.family == "qwen-image":
                if manifest.variant != "original":
                    raise ValueError("Qwen restoration captions require an original-variant cache")
                content = {
                    "target": content["image"],
                    "caption_utf8": content["caption_utf8"],
                    "references": [],
                    "foregrounds": [],
                }
            if manifest.family == "qwen-image-2-1":
                content = {"target": content["image"], "caption_utf8": content["caption_utf8"], "references": []}
            source = canonical_sha256(content)
            identity = validate_latent_entry(cache, manifest, entry, release_cache=True)
            if identity.source_sha256 != source:
                raise ValueError("Restoration source/caption contents differ from their cache identity; rebuild cache")
            pairs.append(AlignedImagePair(None if synthetic else low[path.stem], path, entry, cache, degradation))
            yield {
                "high": source,
                "low": None if synthetic else file_sha256(low[path.stem], release_cache=True),
                "cache": entry.sha256,
            }

    digest = canonical_sequence_sha256(
        records(),
        field="entries",
        context={
            "schema": "restoration-images-random-delivery-3" if synthetic else "restoration-images-2",
            "degradation": None if degradation is None else degradation.model_dump(mode="json"),
        },
    )
    return tuple(pairs), digest
