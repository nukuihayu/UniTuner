"""Store content-addressed native latent/text records and immutable file manifests."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from unitunner.artifacts.atomic import atomic_output
from unitunner.artifacts.hashing import file_sha256
from unitunner.artifacts.tensors import TensorReader, save_tensors
from unitunner.data.cache.schema import CacheIdentity
from unitunner.data.validation import ordered_validation_map


class CacheEntry(BaseModel):
    """Pin one relative content-addressed record and its complete file contents."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    file: str = Field(pattern=r"^[0-9a-f]{64}\.safetensors$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def save_cache_record(
    directory: Path,
    identity: CacheIdentity,
    tensors: Mapping[str, Tensor],
    *,
    metadata: Mapping[str, str] | None = None,
) -> CacheEntry:
    """Publish validated tensors without overwriting an existing content identity.

    Families validate tensor geometry and values before calling. Existing records
    must match identity, keys, dtypes and values exactly. If additional metadata
    is supplied, the complete metadata inventory must also match. Concurrent
    writers to the same identity are unsupported.
    """
    expected_metadata = {**(metadata or {}), "unitunner.cache.identity": identity.model_dump_json()}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{identity.digest()}.safetensors"
    if path.exists():
        with TensorReader(path) as reader:
            stored_metadata = reader.metadata()
            if (
                stored_metadata != expected_metadata
                if metadata is not None
                else stored_metadata.get("unitunner.cache.identity") != expected_metadata["unitunner.cache.identity"]
            ):
                raise ValueError("Cache identity collision: metadata differs")
            if set(reader.keys()) != tensors.keys():
                raise ValueError("Cache identity collision: tensor keys differ")
            for key, value in tensors.items():
                stored = reader.read(key)
                if stored.dtype != value.dtype or not torch.equal(stored, value.detach().cpu()):
                    raise ValueError(
                        "Cache identity collision: numerical contents differ; include sampling policy in identity"
                    )
    else:
        save_tensors(tensors, path, metadata=expected_metadata)
    return CacheEntry(file=path.name, sha256=file_sha256(path))


class LatentManifest(BaseModel):
    """Pin ordered native image latent/text records, independent of training repeats."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal[1] = 1
    family: Literal["z-image", "moss-image", "flux", "flux2", "flux3", "qwen-image", "qwen-image-2-1", "krea2"] = (
        "z-image"
    )
    variant: (
        Literal["dev", "kontext", "fill", "base", "distilled", "original", "edit", "edit-2509", "edit-2511", "layered"]
        | None
    ) = None
    version: Literal["dev", "klein-4b", "klein-9b"] | None = None
    entries: tuple[CacheEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_variant(self) -> Self:
        """Require the exact variant for native Flux and FLUX3 cache records."""
        variants = {
            "moss-image": (None, "edit"),
            "flux": ("dev", "kontext", "fill"),
            "flux3": ("base", "distilled"),
            "flux2": ("base", "distilled"),
            "qwen-image": ("original", "edit", "edit-2509", "edit-2511", "layered"),
        }
        if self.family in variants:
            if self.variant not in variants[self.family]:
                raise ValueError("Cache variant must match its native family")
        elif self.variant is not None:
            raise ValueError("Cache variant must be supplied exactly for Flux/FLUX3/Qwen Image")
        if self.family == "flux2":
            if self.version is None or (self.version == "dev" and self.variant != "distilled"):
                raise ValueError("FLUX.2 cache requires version and a published checkpoint variant")
        elif self.version is not None:
            raise ValueError("Cache version applies only to FLUX.2")
        return self


def _record_keys(layout: str) -> set[str]:
    """Declare exact tensor inventories for supported native cache layouts."""
    qwen21 = re.fullmatch(r"qwen-image-2-1/C1HW\+LD\+R(0|[1-9][0-9]*)", layout)
    if qwen21 is not None:
        return {
            "latents",
            "caption",
            "image_slots",
            "reference_grids",
            "reference_hashes",
            *(f"reference.{index}" for index in range(int(qwen21[1]))),
        }
    qwen = re.fullmatch(r"qwen-image/(original|edit|edit-2509|edit-2511|layered)/CLHW\+LD\+R(0|[1-9][0-9]*)", layout)
    if qwen is not None:
        if qwen[1] in ("original", "layered") and qwen[2] != "0":
            raise ValueError("Qwen original/layered cache cannot contain edit references")
        if qwen[1] == "edit" and qwen[2] == "0":
            raise ValueError("Original Qwen edit cache requires references")
        return {"latents", "caption", *(f"reference.{index}" for index in range(int(qwen[2])))}
    flux2 = re.fullmatch(r"flux2/(dev|klein-4b|klein-9b)/(base|distilled)/CHW\+LD\+R(0|[1-9][0-9]*)", layout)
    if flux2 is not None:
        return {"latents", "caption", *(f"reference.{index}" for index in range(int(flux2[3])))}
    moss = re.fullmatch(r"moss-image/edit/CHW\+LD\+R([1-9][0-9]*)", layout)
    if moss is not None:
        return {"latents", "caption", *(f"reference.{index}" for index in range(int(moss[1])))}
    flux3 = re.fullmatch(r"flux3/(base|distilled)/CHW\+LD\+R(10|[0-9])", layout)
    if flux3 is not None:
        return {"latents", "caption", *(f"reference.{index}" for index in range(int(flux3[2])))}
    match layout:
        case "z-image/CHW+LD" | "moss-image/CHW+LD" | "krea2/C1HW+SLD":
            return {"latents", "caption"}
        case "flux/dev/CHW+LD+V":
            return {"latents", "caption", "pooled"}
        case "flux/kontext/CHW+LD+V":
            return {"latents", "caption", "pooled", "control_latents"}
        case "flux/fill/CHW+LD+V":
            return {"latents", "caption", "pooled", "control_latents", "fill_mask"}
        case _:
            raise ValueError("Image records require a supported native family layout")


@dataclass(frozen=True, slots=True)
class ImageCacheRecord:
    """Carry FP32 CHW/CLHW model latents, LD captions and explicit optional conditions.

    Krea2 uses C1HW latents and compact SLD selected-layer captions.
    Flux pooled is a V vector. Control latents are CHW VAE-normalized values;
    Fill mask is 64xlatent_heightxlatent_width after native 8x8 unshuffle.
    FLUX.2/FLUX3 references are ordered CHW means with independent spatial grids.
    Qwen targets use CLHW and references use C1HW; layers are independent images.
    """

    latents: Tensor
    caption: Tensor
    pooled: Tensor | None = None
    control_latents: Tensor | None = None
    fill_mask: Tensor | None = None
    references: tuple[Tensor, ...] = ()


def save_latent_record(
    directory: Path,
    identity: CacheIdentity,
    latents: Tensor,
    caption: Tensor,
    *,
    pooled: Tensor | None = None,
    control_latents: Tensor | None = None,
    fill_mask: Tensor | None = None,
    references: tuple[Tensor, ...] = (),
) -> CacheEntry:
    """Atomically save native CHW/CLHW latents and nonempty LD text as CPU-readable data.

    Identity must pin source, encoder and transform contents including posterior
    sampling policy. This function preserves values and does not encode pixels.
    The returned digest pins the complete file, including its identity metadata.
    Existing numerical contents are immutable; colliding identities are rejected.
    Concurrent writers to the same identity are unsupported.
    """
    krea2 = identity.layout == "krea2/C1HW+SLD"
    expected = _record_keys(identity.layout)
    tensors = {"latents": latents, "caption": caption}
    tensors.update({f"reference.{index}": value for index, value in enumerate(references)})
    for key, value in (("pooled", pooled), ("control_latents", control_latents), ("fill_mask", fill_mask)):
        if value is not None:
            tensors[key] = value
    if tensors.keys() != expected:
        raise ValueError("Cache record tensors differ from the native layout")
    for key, value in tensors.items():
        dimensions = (
            1
            if key == "pooled"
            else (3 if krea2 else 2)
            if key == "caption"
            else 4
            if krea2 or identity.layout.startswith("qwen-image/")
            else 3
        )
        if (
            value.ndim != dimensions
            or not value.numel()
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Cache record requires finite nonempty native tensor: {key}")
    if krea2:
        _validate_krea2_record(latents, caption)
    if identity.layout.startswith("qwen-image/"):
        layered = identity.layout.startswith("qwen-image/layered/")
        if any(value % 2 for value in latents.shape[-2:]) or (
            latents.shape[1] < 2 if layered else latents.shape[1] != 1
        ):
            raise ValueError("Qwen cache requires even latent H/W and native target layer count")
        if any(value.shape[1] != 1 or any(size % 2 for size in value.shape[-2:]) for value in references):
            raise ValueError("Qwen reference cache requires C1HW and even latent H/W")
    if fill_mask is not None and (
        fill_mask.shape[0] != 64
        or fill_mask.shape[1:] != latents.shape[1:]
        or (fill_mask < 0).any()
        or (fill_mask > 1).any()
    ):
        raise ValueError("Fill cache mask must be 64HW in [0,1] on the target grid")
    if control_latents is not None and (
        control_latents.shape[0] != latents.shape[0]
        or (fill_mask is not None and control_latents.shape != latents.shape)
    ):
        raise ValueError("Cache condition latent channels/grids differ from targets")
    if any(value.shape[0] != latents.shape[0] for value in references):
        raise ValueError("Reference cache latent channels differ from targets")
    return save_cache_record(directory, identity, tensors)


def save_latent_manifest(manifest: LatentManifest, directory: Path) -> Path:
    """Publish an ordered manifest after checking every referenced file's digest."""
    for entry in manifest.entries:
        if file_sha256(directory / entry.file) != entry.sha256:
            raise ValueError("Latent cache file differs from its manifest digest")
    path = directory / "manifest.json"
    with atomic_output(path) as temporary:
        temporary.write_text(manifest.model_dump_json(indent=2) + "\n")
    return path


def read_latent_manifest(directory: Path) -> LatentManifest:
    """Parse the manifest only; callers must validate each record before use."""
    return LatentManifest.model_validate_json((directory / "manifest.json").read_bytes())


def validate_latent_entry(
    directory: Path, manifest: LatentManifest, entry: CacheEntry, *, release_cache: bool = False
) -> CacheIdentity:
    """Validate file content and native headers without loading tensor payloads."""
    path = directory / entry.file
    if file_sha256(path, release_cache=release_cache) != entry.sha256:
        raise ValueError("Latent cache file differs from its manifest digest")
    with TensorReader(path) as reader:
        encoded = reader.metadata().get("unitunner.cache.identity")
        if encoded is None:
            raise ValueError("Latent cache record is missing its content identity")
        identity = CacheIdentity.model_validate_json(encoded)
        layout = f"flux/{manifest.variant}/CHW+LD+V" if manifest.family == "flux" else f"{manifest.family}/CHW+LD"
        if manifest.family == "krea2":
            layout = "krea2/C1HW+SLD"
        if manifest.family == "moss-image" and manifest.variant == "edit":
            count = sum(key.startswith("reference.") for key in reader.keys())
            layout = f"moss-image/edit/CHW+LD+R{count}"
        if manifest.family == "flux2":
            count = sum(key.startswith("reference.") for key in reader.keys())
            layout = f"flux2/{manifest.version}/{manifest.variant}/CHW+LD+R{count}"
        if manifest.family == "flux3":
            count = sum(key.startswith("reference.") for key in reader.keys())
            layout = f"flux3/{manifest.variant}/CHW+LD+R{count}"
        if manifest.family == "qwen-image":
            count = sum(key.startswith("reference.") for key in reader.keys())
            layout = f"qwen-image/{manifest.variant}/CLHW+LD+R{count}"
        if manifest.family == "qwen-image-2-1":
            count = sum(key.startswith("reference.") for key in reader.keys())
            layout = f"qwen-image-2-1/C1HW+LD+R{count}"
        if set(reader.keys()) != _record_keys(layout):
            raise ValueError("Latent cache record keys differ from native family layout")
        if identity.layout != layout or identity.digest() + ".safetensors" != entry.file:
            raise ValueError("Latent cache identity differs from its layout or filename")
    return identity


def load_latent_manifest(directory: Path, *, validation_workers: int = 4) -> LatentManifest:
    """Preflight immutable cache files once before entering the training hot loop."""
    manifest = read_latent_manifest(directory)
    def validate(entry: CacheEntry) -> None:
        validate_latent_entry(directory, manifest, entry, release_cache=True)

    for _ in ordered_validation_map(validate, manifest.entries, workers=validation_workers):
        pass
    return manifest


@dataclass(frozen=True, slots=True)
class CaptionCacheRecord:
    """Borrow only the text conditions of an already preflighted immutable image record."""

    caption: Tensor
    pooled: Tensor | None = None


def read_caption_record(directory: Path, entry: CacheEntry, *, device: torch.device) -> CaptionCacheRecord:
    """Read restoration text without faulting or transferring unused image payloads.

    Indexing must validate the immutable record first. Native key inventories
    and text geometry are checked here; latents/references remain on disk.
    """
    with TensorReader(directory / entry.file) as reader:
        encoded = reader.metadata().get("unitunner.cache.identity")
        if encoded is None:
            raise ValueError("Caption cache record is missing its content identity")
        identity = CacheIdentity.model_validate_json(encoded)
        if identity.digest() + ".safetensors" != entry.file or set(reader.keys()) != _record_keys(identity.layout):
            raise ValueError("Caption cache identity differs from its keys or filename")
        caption = reader.read("caption")
        krea2 = identity.layout == "krea2/C1HW+SLD"
        if caption.ndim != (3 if krea2 else 2) or min(caption.shape) < 1 or not caption.is_floating_point():
            raise ValueError("Invalid cached caption dimensions/dtype")
        if (krea2 or identity.layout.startswith("qwen-image-2-1/")) and (
            caption.dtype != torch.float32 or not torch.isfinite(caption).all()
        ):
            raise ValueError("Krea2/Qwen Image2.1 caption requires finite FP32 values")
        pooled = reader.read("pooled", device=device, dtype=torch.float32) if "pooled" in reader.keys() else None
        return CaptionCacheRecord(caption.to(device=device, dtype=torch.float32), pooled)


def read_latent_record(directory: Path, entry: CacheEntry, *, device: torch.device) -> ImageCacheRecord:
    """Read native FP32 tensors on the explicit training device after manifest preflight.

    Files must remain immutable after preflight. CPU tensors may share private
    mmap storage until stacked; numerical inputs are not rescaled or repacked.
    """
    with TensorReader(directory / entry.file) as reader:
        encoded = reader.metadata().get("unitunner.cache.identity")
        if encoded is None:
            raise ValueError("Latent cache record is missing its content identity")
        identity = CacheIdentity.model_validate_json(encoded)
        krea2 = identity.layout == "krea2/C1HW+SLD"
        values = {
            key: reader.read(key) if krea2 else reader.read(key, device=device, dtype=torch.float32)
            for key in reader.keys()
        }
        if krea2:
            if set(values) != {"latents", "caption"}:
                raise ValueError("Krea2 record requires exactly latents and caption")
            _validate_krea2_record(values["latents"], values["caption"])
            values = {key: value.to(device) for key, value in values.items()}
    latents, caption = values["latents"], values["caption"]
    if (
        latents.ndim != (4 if krea2 or identity.layout.startswith("qwen-image/") else 3)
        or caption.ndim != (3 if krea2 else 2)
        or min(*latents.shape, *caption.shape) <= 0
    ):
        raise ValueError("Invalid cached latent/text dimensions")
    return ImageCacheRecord(
        latents,
        caption,
        values.get("pooled"),
        values.get("control_latents"),
        values.get("fill_mask"),
        tuple(values[f"reference.{index}"] for index in range(sum(key.startswith("reference.") for key in values))),
    )


def _validate_krea2_record(latents: Tensor, caption: Tensor) -> None:
    """Validate native FP32 latent16 single-frame geometry and compact selected text."""
    if (
        latents.ndim != 4
        or latents.shape[:2] != (16, 1)
        or min(latents.shape) < 1
        or any(size % 2 for size in latents.shape[-2:])
        or caption.ndim != 3
        or min(caption.shape) < 1
        or any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in (latents, caption))
    ):
        raise ValueError("Krea2 cache requires finite FP32 16x1xHxW with even H/W and nonempty SLD caption")
