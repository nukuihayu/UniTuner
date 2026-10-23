"""Persist unified Qwen2.1 VAE/text records with exact reference geometry and hashes."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from unitunner.artifacts.tensors import TensorReader
from unitunner.data.cache.schema import CacheIdentity
from unitunner.data.cache.storage import CacheEntry, save_cache_record


@dataclass(frozen=True, slots=True)
class QwenImage21CacheRecord:
    """Carry FP32 64x1xHxW target/references, LD text and ordered reference metadata.

    Image slots are R int64 retained-text offsets; grids are R2 int64 (H,W).
    Hashes are R32 uint8 SHA256 of the exact RGBA media shared by VAE/vision.
    One immutable record publishes both encodings, eliminating stale split caches.
    """

    latents: Tensor
    caption: Tensor
    references: tuple[Tensor, ...]
    image_slots: Tensor
    reference_grids: Tensor
    reference_hashes: Tensor

    def validate(self) -> None:
        """Reject nonnative dimensions/dtypes/values and inconsistent reference metadata."""
        for value in (self.latents, *self.references):
            if (
                value.ndim != 4
                or value.shape[:2] != (64, 1)
                or min(value.shape) < 1
                or any(size % 2 for size in value.shape[-2:])
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Qwen Image2.1 cache requires finite FP32 64x1xHxW with even H/W")
        if (
            self.caption.ndim != 2
            or min(self.caption.shape) < 1
            or self.caption.dtype != torch.float32
            or not torch.isfinite(self.caption).all()
        ):
            raise ValueError("Qwen Image2.1 caption requires finite nonempty FP32 LD")
        count = len(self.references)
        if self.image_slots.dtype != torch.int64 or self.image_slots.shape != (count,):
            raise ValueError("Qwen Image2.1 image_slots requires R int64")
        if self.reference_grids.dtype != torch.int64 or self.reference_grids.shape != (count, 2):
            raise ValueError("Qwen Image2.1 reference_grids requires R2 int64")
        if self.reference_hashes.dtype != torch.uint8 or self.reference_hashes.shape != (count, 32):
            raise ValueError("Qwen Image2.1 reference_hashes requires R32 uint8")
        slots = self.image_slots.cpu().tolist()
        if slots != sorted(slots) or any(slot < 0 or slot > self.caption.shape[0] for slot in slots):
            raise ValueError("Qwen Image2.1 image_slots requires ordered offsets within text length")
        grids = self.reference_grids.cpu().tolist()
        if grids != [list(value.shape[-2:]) for value in self.references]:
            raise ValueError("Qwen Image2.1 vae/vision reference grids differ")


def save_qwen_image_21_record(directory: Path, identity: CacheIdentity, record: QwenImage21CacheRecord) -> CacheEntry:
    """Validate and atomically publish one content-addressed unified record.

    Existing identities are immutable: metadata, dtypes and every tensor value
    must be exact. Caller owns encoder/source/transform digests. No mode or RNG
    mutation occurs; tensors are detached and persisted as CPU-readable data.
    """
    record.validate()
    if identity.layout != f"qwen-image-2-1/C1HW+LD+R{len(record.references)}":
        raise ValueError("Qwen Image2.1 identity layout differs from reference count")
    values = {
        "latents": record.latents,
        "caption": record.caption,
        "image_slots": record.image_slots,
        "reference_grids": record.reference_grids,
        "reference_hashes": record.reference_hashes,
    }
    values.update({f"reference.{i}": value for i, value in enumerate(record.references)})
    return save_cache_record(directory, identity, values)


def read_qwen_image_21_record(directory: Path, entry: CacheEntry, *, device: torch.device) -> QwenImage21CacheRecord:
    """Read exact native tensor dtypes after manifest preflight and validate metadata.

    Files must remain immutable after preflight. Floating features are not
    rescaled; integer slot/grid/hash tensors retain their exact storage dtypes
    on the host. Validate CPU file contents before moving floating inputs, so
    training does not synchronize the device to read per-token metadata.
    """
    with TensorReader(directory / entry.file) as reader:
        encoded = reader.metadata().get("unitunner.cache.identity")
        if encoded is None:
            raise ValueError("Qwen Image2.1 record lacks content identity")
        identity = CacheIdentity.model_validate_json(encoded)
        count = sum(key.startswith("reference.") for key in reader.keys())
        keys = {
            "latents",
            "caption",
            "image_slots",
            "reference_grids",
            "reference_hashes",
            *(f"reference.{i}" for i in range(count)),
        }
        if (
            set(reader.keys()) != keys
            or identity.layout != f"qwen-image-2-1/C1HW+LD+R{count}"
            or identity.digest() + ".safetensors" != entry.file
        ):
            raise ValueError("Qwen Image2.1 record keys/layout/filename differ")
        values = {key: reader.read(key) for key in keys}
    record = QwenImage21CacheRecord(
        values["latents"],
        values["caption"],
        tuple(values[f"reference.{i}"] for i in range(count)),
        values["image_slots"],
        values["reference_grids"],
        values["reference_hashes"],
    )
    record.validate()
    return QwenImage21CacheRecord(
        record.latents.to(device),
        record.caption.to(device),
        tuple(value.to(device) for value in record.references),
        record.image_slots,
        record.reference_grids,
        record.reference_hashes,
    )
