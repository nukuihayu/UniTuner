"""Immutable single-frame H3-Image caches, separate from H3 audio/video records."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from unituner.artifacts.atomic import atomic_output
from unituner.artifacts.hashing import file_sha256
from unituner.artifacts.tensors import TensorReader
from unituner.data.cache.schema import CacheIdentity
from unituner.data.cache.storage import CacheEntry, save_cache_record
from unituner.data.validation import ordered_validation_map


class MiniMaxH3ImageManifest(BaseModel):
    """Pin ordered image records and their independent text/vision layout."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal[1] = 1
    family: Literal["minimax-h3-image"] = "minimax-h3-image"
    variant: Literal["base"] = "base"
    entries: tuple[CacheEntry, ...] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class MiniMaxH3ImageRecord:
    latents: Tensor  # 24x1xHxW
    text: Tensor  # Lx5120
    token_tags: Tensor  # L, int64
    references: tuple[Tensor, ...] = ()  # ordered 24x1xHxW


def _layout(count: int) -> str:
    if count < 0 or count > 16:
        raise ValueError("H3-Image supports up to 16 references")
    return f"minimax-h3-image/1/C1HW+LD+tags+R{count}"


def _validate(record: MiniMaxH3ImageRecord) -> None:
    if (
        record.latents.ndim != 4
        or record.latents.shape[:2] != (24, 1)
        or any(size < 2 or size % 2 for size in record.latents.shape[-2:])
        or record.text.ndim != 2
        or record.text.shape[1] != 5120
        or record.token_tags.shape != (record.text.shape[0],)
        or record.token_tags.dtype != torch.int64
        or not torch.all((record.token_tags == 0) | (record.token_tags == 1))
    ):
        raise ValueError("H3-Image cache needs 24x1xHxW latents, Lx5120 text and L binary tags")
    _layout(len(record.references))
    for ref in record.references:
        if ref.ndim != 4 or ref.shape[:2] != (24, 1) or any(size < 2 or size % 2 for size in ref.shape[-2:]):
            raise ValueError("H3-Image reference cache requires even 24x1xHxW latents")
    if any(
        value.dtype != torch.float32 or not torch.isfinite(value).all()
        for value in (record.latents, record.text, *record.references)
    ):
        raise ValueError("H3-Image cache values must be finite FP32")


def save_image_record(directory: Path, identity: CacheIdentity, record: MiniMaxH3ImageRecord) -> CacheEntry:
    _validate(record)
    if identity.layout != _layout(len(record.references)):
        raise ValueError("H3-Image cache identity has the wrong reference layout")
    return save_cache_record(
        directory,
        identity,
        {
            "latents": record.latents,
            "text": record.text,
            "token_tags": record.token_tags,
            **{f"reference.{index}": value for index, value in enumerate(record.references)},
        },
    )


def save_image_manifest(directory: Path, manifest: MiniMaxH3ImageManifest) -> None:
    for entry in manifest.entries:
        validate_image_entry(directory, entry)
    with atomic_output(directory / "manifest.json") as temporary:
        temporary.write_text(manifest.model_dump_json(indent=2) + "\n")


def validate_image_entry(directory: Path, entry: CacheEntry) -> CacheIdentity:
    if file_sha256(directory / entry.file, release_cache=True) != entry.sha256:
        raise ValueError("H3-Image record differs from its manifest digest")
    with TensorReader(directory / entry.file) as reader:
        encoded = reader.metadata().get("unituner.cache.identity")
        if encoded is None:
            raise ValueError("H3-Image record is missing its content identity")
        identity = CacheIdentity.model_validate_json(encoded)
        count = sum(key.startswith("reference.") for key in reader.keys())
        keys = {"latents", "text", "token_tags", *(f"reference.{i}" for i in range(count))}
        if (
            identity.layout != _layout(count)
            or entry.file != identity.digest() + ".safetensors"
            or set(reader.keys()) != keys
        ):
            raise ValueError("H3-Image record layout, keys or filename differ")
    return identity


def load_image_manifest(directory: Path, *, validation_workers: int = 4) -> MiniMaxH3ImageManifest:
    manifest = MiniMaxH3ImageManifest.model_validate_json((directory / "manifest.json").read_bytes())
    for _ in ordered_validation_map(
        lambda entry: validate_image_entry(directory, entry), manifest.entries, workers=validation_workers
    ):
        pass
    return manifest


def read_image_record(directory: Path, entry: CacheEntry, *, device: torch.device) -> MiniMaxH3ImageRecord:
    with TensorReader(directory / entry.file) as reader:
        count = sum(key.startswith("reference.") for key in reader.keys())
        record = MiniMaxH3ImageRecord(
            reader.read("latents", device=device, dtype=torch.float32),
            reader.read("text", device=device, dtype=torch.float32),
            reader.read("token_tags").to(device=device),
            tuple(reader.read(f"reference.{i}", device=device, dtype=torch.float32) for i in range(count)),
        )
    _validate(record)
    return record
