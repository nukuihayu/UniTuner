"""Store immutable native Wan latent/text/image/control records and exact variant manifests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from unituner.artifacts.atomic import atomic_output
from unituner.artifacts.hashing import file_sha256
from unituner.artifacts.tensors import TensorReader
from unituner.data.cache.schema import CacheIdentity
from unituner.data.cache.storage import CacheEntry, save_cache_record


class WanCacheFormat(BaseModel):
    """Pin Wan variant, frame layout, precision and condition geometry independently of records."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal[1] = 1
    family: Literal["wan"] = "wan"
    version: Literal["2.1", "2.2"]
    variant: Literal["t2v", "i2v"]
    text_dim: int = Field(gt=0)
    image_dim: int | None = Field(default=None, gt=0)
    control: bool = False
    frame_layout: Literal["video", "single-frame"] = "video"
    clean_count: int = Field(default=0, ge=0, le=2)

    @model_validator(mode="after")
    def validate_image_features(self) -> Self:
        """Require CLIP features exactly for the Wan2.1 I2V branch."""
        if (self.image_dim is not None) != (self.version == "2.1" and self.variant == "i2v"):
            raise ValueError("Wan image features are required exactly for2.1 I2V")
        if self.frame_layout == "single-frame":
            if self.variant != "i2v" or self.control or self.clean_count not in (1, 2):
                raise ValueError("Wan single-frame cache requires I2V, one/two clean frames and no control video")
        elif self.clean_count:
            raise ValueError("Wan clean_count requires single-frame layout")
        return self

    @property
    def layout(self) -> str:
        """Identify CFHW records with explicit temporal layout and optional conditions."""
        if self.frame_layout == "single-frame":
            return f"wan/{self.version}/{self.variant}/single-frame/CFHW+LD+I{int(self.image_dim is not None)}+K{self.clean_count}"
        return f"wan/{self.version}/{self.variant}/CFHW+LD+I{int(self.image_dim is not None)}+C{int(self.control)}"


class WanCacheManifest(WanCacheFormat):
    """Pin a nonempty ordered list of complete content-addressed Wan records."""

    entries: tuple[CacheEntry, ...] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class WanCacheRecord:
    """Carry one window's FP32 16FHW means, LD text and native optional conditions."""

    latents: Tensor
    caption: Tensor
    image_condition: Tensor | None = None
    image_features: Tensor | None = None
    control_latents: Tensor | None = None
    frame_indices: tuple[int, ...] | None = None
    target_position: int | None = None


def _record_tensors(record: WanCacheRecord, manifest: WanCacheFormat) -> dict[str, Tensor]:
    """Validate native condition inventory, temporal/spatial grids and finite FP32 values."""
    latents, caption = record.latents, record.caption
    tensors = {"latents": latents, "caption": caption}
    for name, value in (
        ("image_condition", record.image_condition),
        ("image_features", record.image_features),
        ("control_latents", record.control_latents),
    ):
        if value is not None:
            tensors[name] = value
    single_frame = manifest.frame_layout == "single-frame"
    if single_frame:
        indices, position = record.frame_indices, record.target_position
        if (
            indices is None
            or len(indices) != manifest.clean_count + 1
            or any(type(index) is not int or not 0 <= index < 1024 for index in indices)
            or tuple(sorted(set(indices))) != indices
            or type(position) is not int
            or not 0 <= position < len(indices)
        ):
            raise ValueError("Wan single-frame record requires sorted unique frame_indices and a valid target_position")
        tensors["frame_indices"] = torch.tensor(indices, dtype=torch.int64)
        tensors["target_position"] = torch.tensor(position, dtype=torch.int64)
    elif record.frame_indices is not None or record.target_position is not None:
        raise ValueError("Wan explicit temporal indices require single-frame layout")
    if (
        latents.ndim != 4
        or latents.shape[0] != 16
        or min(latents.shape) < 1
        or any(size % 2 for size in latents.shape[-2:])
        or caption.ndim != 2
        or not 1 <= caption.shape[0] <= 512
        or caption.shape[1] != manifest.text_dim
    ):
        raise ValueError("Wan cache requires16FHW means with even spatial grids and nonempty LD text")
    for name, present in (
        ("image_condition", manifest.variant == "i2v"),
        ("image_features", manifest.image_dim is not None),
        ("control_latents", manifest.control),
    ):
        if (name in tensors) != present:
            raise ValueError(f"Wan cache condition inventory differs from manifest: {name}")
    if record.image_condition is not None:
        condition = record.image_condition
        if condition.shape != (20, *latents.shape[1:]):
            raise ValueError("Wan I2V condition must be20FHW on the target grid")
        if single_frame:
            if latents.shape[1] != manifest.clean_count + 1:
                raise ValueError("Wan single-frame latent count differs from clean/target coordinates")
            for index in range(latents.shape[1]):
                visible = 0 if index == record.target_position else 1
                if not torch.equal(condition[:4, index], torch.full_like(condition[:4, index], visible)):
                    raise ValueError("Wan single-frame visibility differs from target_position")
                if visible and not torch.equal(condition[4:, index], latents[:, index]):
                    raise ValueError("Wan single-frame clean condition differs from its training latent")
        elif not torch.equal(condition[:4, :1], torch.ones_like(condition[:4, :1])) or torch.count_nonzero(
            condition[:4, 1:]
        ):
            raise ValueError("Wan regular I2V first-frame visibility is invalid")
    image_shape = (manifest.clean_count, 257, manifest.image_dim) if single_frame else (257, manifest.image_dim)
    if record.image_features is not None and record.image_features.shape != image_shape:
        raise ValueError("Wan CLIP features require native clean-frame count,257 tokens and the declared width")
    if record.control_latents is not None and record.control_latents.shape != latents.shape:
        raise ValueError("Wan regular control latents must match the target window grid")
    if any(
        value.dtype != torch.float32 or not torch.isfinite(value).all()
        for name, value in tensors.items()
        if name not in ("frame_indices", "target_position")
    ):
        raise ValueError("Wan cache tensors must be finite FP32")
    return tensors


def save_wan_record(
    directory: Path, identity: CacheIdentity, record: WanCacheRecord, *, manifest: WanCacheFormat
) -> CacheEntry:
    """Atomically save exact native tensors; reject content-address collisions without overwriting."""
    if identity.layout != manifest.layout:
        raise ValueError("Wan cache identity layout differs from manifest")
    tensors = _record_tensors(record, manifest)
    return save_cache_record(directory, identity, tensors)


def read_wan_record(
    directory: Path, entry: CacheEntry, *, manifest: WanCacheManifest, device: torch.device
) -> WanCacheRecord:
    """Read immutable owned native FP32 values after manifest preflight; never rescale or repack."""
    with TensorReader(directory / entry.file) as reader:
        encoded = reader.metadata().get("unituner.cache.identity")
        if encoded is None:
            raise ValueError("Wan cache record has no content identity")
        identity = CacheIdentity.model_validate_json(encoded)
        if identity.layout != manifest.layout or identity.digest() + ".safetensors" != entry.file:
            raise ValueError("Wan record identity differs from its manifest layout or filename")
        values = {name: reader.read(name) for name in reader.keys()}
    expected = {"latents", "caption"}
    if manifest.variant == "i2v":
        expected.add("image_condition")
    if manifest.image_dim is not None:
        expected.add("image_features")
    if manifest.control:
        expected.add("control_latents")
    if manifest.frame_layout == "single-frame":
        expected.update(("frame_indices", "target_position"))
    if values.keys() != expected:
        raise ValueError("Wan cache keys differ from manifest")
    indices, position = values.get("frame_indices"), values.get("target_position")
    if indices is not None and (
        indices.dtype != torch.int64
        or indices.ndim != 1
        or position is None
        or position.dtype != torch.int64
        or position.shape != ()
    ):
        raise ValueError("Wan frame_indices/target_position require native int64 vector/scalar storage")
    record = WanCacheRecord(
        values["latents"],
        values["caption"],
        values.get("image_condition"),
        values.get("image_features"),
        values.get("control_latents"),
        None if indices is None else tuple(indices.tolist()),
        None if position is None else int(position.item()),
    )
    _record_tensors(record, manifest)
    return WanCacheRecord(
        record.latents.to(device=device, copy=True),
        record.caption.to(device=device, copy=True),
        None if record.image_condition is None else record.image_condition.to(device=device, copy=True),
        None if record.image_features is None else record.image_features.to(device=device, copy=True),
        None if record.control_latents is None else record.control_latents.to(device=device, copy=True),
        record.frame_indices,
        record.target_position,
    )


def save_wan_manifest(manifest: WanCacheManifest, directory: Path) -> Path:
    """Publish a complete ordered manifest only after every record digest is verified."""
    for entry in manifest.entries:
        if file_sha256(directory / entry.file) != entry.sha256:
            raise ValueError("Wan cache record differs from manifest digest")
    path = directory / "manifest.json"
    with atomic_output(path) as temporary:
        temporary.write_text(manifest.model_dump_json(indent=2) + "\n")
    return path


def load_wan_manifest(directory: Path) -> WanCacheManifest:
    """Preflight all complete file identities and tensor contracts before training or cache reuse."""
    manifest = WanCacheManifest.model_validate_json((directory / "manifest.json").read_text())
    for entry in manifest.entries:
        if file_sha256(directory / entry.file) != entry.sha256:
            raise ValueError("Wan cache record differs from manifest digest")
        read_wan_record(directory, entry, manifest=manifest, device=torch.device("cpu"))
    return manifest
