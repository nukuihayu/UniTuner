"""Publish and verify frozen encoder target grids keyed by complete native latent records."""

import hashlib
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from unituner.artifacts.atomic import atomic_output
from unituner.artifacts.hashing import canonical_sha256, file_sha256
from unituner.artifacts.tensors import TensorReader, save_tensors
from unituner.config.features import RepaConfig
from unituner.data.cache.storage import CacheEntry


class RepaEntry(BaseModel):
    """Pin one native record digest and the complete frozen THWE target artifact."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact: CacheEntry


class RepaManifest(BaseModel):
    """Pin frozen encoder identity and all target artifacts, preserving native record ownership."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal[1] = 1
    encoder_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_dim: int = Field(gt=0)
    entries: tuple[RepaEntry, ...] = Field(min_length=1)


def save_repa_target(directory: Path, source: CacheEntry, features: Tensor, *, encoder_sha256: str) -> RepaEntry:
    """Atomically store finite frozen FP32 THWE features from the native target's same crop/frames.

    The producer supplies real encoder features, excludes CLS/register tokens,
    and owns preprocessing. Detach the encoder graph; no target is synthesized
    from latents. The complete native record digest binds its crop/time identity.
    """
    if (
        features.ndim != 4
        or min(features.shape) < 1
        or not features.is_floating_point()
        or not torch.isfinite(features).all()
    ):
        raise ValueError("REPA targets require finite nonempty THWE encoder features")
    # Validate identities before creating a destination.
    RepaManifest(
        encoder_sha256=encoder_sha256,
        encoder_dim=features.shape[-1],
        entries=(RepaEntry(source_sha256=source.sha256, artifact=source),),
    )
    values = features.detach().cpu().float().contiguous()
    content_digest = hashlib.sha256(memoryview(values.numpy())).hexdigest()
    digest = canonical_sha256(
        {"source": source.sha256, "encoder": encoder_sha256, "shape": list(values.shape), "features": content_digest}
    )
    path = directory / (digest + ".safetensors")
    save_tensors(
        {"features": values},
        path,
        metadata={"repa.source_sha256": source.sha256, "repa.encoder_sha256": encoder_sha256},
    )
    return RepaEntry(source_sha256=source.sha256, artifact=CacheEntry(file=path.name, sha256=file_sha256(path)))


def save_repa_manifest(manifest: RepaManifest, directory: Path) -> None:
    """Validate unique complete targets and atomically publish their explicit JSON inventory."""
    if len({entry.source_sha256 for entry in manifest.entries}) != len(manifest.entries):
        raise ValueError("REPA manifest source digests must be unique")
    if any(file_sha256(directory / entry.artifact.file) != entry.artifact.sha256 for entry in manifest.entries):
        raise ValueError("REPA target file digest differs from manifest")
    with atomic_output(directory / "manifest.json") as temporary:
        temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


class RepaTargets:
    """Verify a target cache once, then read complete CPU grids in native sample order."""

    def __init__(self, config: RepaConfig) -> None:
        self.directory: Path = Path(config.cache_directory)
        manifest = RepaManifest.model_validate_json((self.directory / "manifest.json").read_text())
        if manifest.encoder_sha256 != config.encoder_sha256 or manifest.encoder_dim != config.encoder_dim:
            raise ValueError("REPA cache encoder identity/dimension differs from method.repa")
        self.entries: dict[str, CacheEntry] = {entry.source_sha256: entry.artifact for entry in manifest.entries}
        if len(self.entries) != len(manifest.entries):
            raise ValueError("REPA cache source digests must be unique")
        self.encoder_sha256: str = config.encoder_sha256
        self.encoder_dim: int = config.encoder_dim
        self.digest: str = file_sha256(self.directory / "manifest.json")
        for entry in self.entries.values():
            if file_sha256(self.directory / entry.file) != entry.sha256:
                raise ValueError("REPA cache target content differs from manifest")

    def read(self, source: CacheEntry) -> Tensor:
        """Read one finite FP32 THWE target, rejecting changed artifacts or wrong native identity."""
        if source.sha256 not in self.entries:
            raise ValueError("REPA cache has no target for this complete native record digest")
        entry = self.entries[source.sha256]
        path = self.directory / entry.file
        if file_sha256(path) != entry.sha256:
            raise ValueError("REPA target changed after cache validation")
        with TensorReader(path) as reader:
            metadata = reader.metadata()
            if (
                set(reader.keys()) != {"features"}
                or metadata.get("repa.source_sha256") != source.sha256
                or metadata.get("repa.encoder_sha256") != self.encoder_sha256
            ):
                raise ValueError("REPA target inventory or encoder/source identity differs")
            features = reader.read("features")
        if (
            features.ndim != 4
            or min(features.shape) < 1
            or features.dtype != torch.float32
            or features.shape[-1] != self.encoder_dim
            or not torch.isfinite(features).all()
        ):
            raise ValueError("REPA cache requires finite FP32 THWE target grids")
        return features

    def validate_sources(self, entries: tuple[CacheEntry, ...]) -> None:
        """Reject a missing target for any native dataset entry before loading model weights."""
        if any(entry.sha256 not in self.entries for entry in entries):
            raise ValueError("REPA cache does not cover every complete native dataset record")


def representation_dataset_sha256(native_sha256: str, targets: RepaTargets | None) -> str:
    """Pin the complete feature manifest into resume identity while preserving non-REPA identities."""
    return native_sha256 if targets is None else canonical_sha256({"native": native_sha256, "repa": targets.digest})
