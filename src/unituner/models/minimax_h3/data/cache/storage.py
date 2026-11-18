"""Store complete immutable H3 AV/text records with explicit reference and timeline geometry."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from unituner.artifacts.atomic import atomic_output
from unituner.artifacts.hashing import file_sha256
from unituner.artifacts.tensors import TensorReader
from unituner.data.cache.schema import CacheIdentity
from unituner.data.cache.storage import CacheEntry, save_cache_record
from unituner.models.minimax_h3.layout import (
    FRAME_RESCALE,
    MiniMaxH3PackedLayout,
    MiniMaxH3ReferenceGeometry,
    MiniMaxH3ReferenceKind,
    MiniMaxH3Task,
    MiniMaxH3TimeOverrides,
    MiniMaxH3VideoGeometry,
    build_layout,
    one_frame_condition_roles,
    reference_condition_role,
    video_latent_frames,
)
from unituner.models.minimax_h3.text.presentation import validate_text_rows


class MiniMaxH3CacheFormat(BaseModel):
    """Pin task, temporal layout and corrected sample-window repair accounting."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal[1] = 1
    family: Literal["minimax-h3"] = "minimax-h3"
    task: MiniMaxH3Task
    frame_layout: Literal["video", "single-frame"] = "video"
    audio_window_accounting: Literal["filled-window-intersection"] = "filled-window-intersection"

    @property
    def layout(self) -> str:
        """Identify native FP32 AV/text values with versioned sample repair semantics."""
        return f"minimax-h3/1/{self.task}/{self.frame_layout}/CFHW+CST+LD+tags/{self.audio_window_accounting}"


class MiniMaxH3CacheManifest(MiniMaxH3CacheFormat):
    """Pin ordered complete AV/text records, independent of repeats and output paths."""

    entries: tuple[CacheEntry, ...] = Field(min_length=1)


class MiniMaxH3CacheReference(BaseModel):
    """Describe one ordered reference; has_audio marks a video's additional audio rows."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    kind: MiniMaxH3ReferenceKind
    has_audio: bool = False

    @model_validator(mode="after")
    def validate_track(self) -> Self:
        """Reserve the optional track flag for video references."""
        if self.kind != "video" and self.has_audio:
            raise ValueError("MiniMaxH3 has_audio applies only to video references")
        return self


type FrameIndex = Annotated[int, Field(ge=0)]


class MiniMaxH3CacheItem(BaseModel):
    """Describe one target window and its ordered conditions in native24Hz pixel units.

    One-frame indices remain nonnegative and need not be sorted or unique; their
    order defines FL2VA slots. Reference blocks have their own native time grid.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    frame_count: int = Field(gt=0)
    crop_start_frame: int = Field(default=0, ge=0)
    audio_present: bool = True
    references: tuple[MiniMaxH3CacheReference, ...] = Field(default=(), max_length=12)
    condition_roles: tuple[Literal["first", "last"], ...] = ()
    target_index: int | None = Field(default=None, ge=0)
    control_indices: tuple[FrameIndex, ...] = ()


@dataclass(frozen=True, slots=True)
class MiniMaxH3CacheRecord:
    """Borrow unbatched FP32 video24FHW/audio32ST, LD text and int64 modality tags.

    S=2 stereo. Text width is5120. Conditions are ordered separately by modality
    using the item's reference order or FL slots. No RCM masks are stored.
    """

    item: MiniMaxH3CacheItem
    video: Tensor
    audio: Tensor
    text: Tensor
    token_tags: Tensor
    visual_conditions: tuple[Tensor, ...] = ()
    audio_conditions: tuple[Tensor, ...] = ()


def _video_geometry(value: Tensor) -> MiniMaxH3VideoGeometry:
    if value.ndim != 4 or value.shape[0] != 24:
        raise ValueError("MiniMaxH3 cached video requires24FHW")
    return MiniMaxH3VideoGeometry(*value.shape[1:])


def _audio_frames(value: Tensor) -> int:
    if value.ndim != 3 or value.shape[:2] != (32, 2) or value.shape[-1] < 1:
        raise ValueError("MiniMaxH3 cached audio requires32x2xT")
    return value.shape[-1]


def build_record_layout(record: MiniMaxH3CacheRecord, *, format: MiniMaxH3CacheFormat) -> MiniMaxH3PackedLayout:
    """Validate complete target/condition geometry and build its native packed row order.

    This is a cache boundary, so finite FP32 values are checked once before use.
    Modality tags are int64 text=1/vision=0 labels, never loss or RCM masks.
    Inputs are borrowed without mutation, casting, rescaling or RNG use.
    """
    item = record.item
    single = format.frame_layout == "single-frame"
    video = _video_geometry(record.video)
    audio_frames = _audio_frames(record.audio)
    if video.frames != (1 if single else video_latent_frames(item.frame_count)) or (single and item.frame_count != 1):
        raise ValueError("MiniMaxH3 cached target frame_count differs from latent geometry")
    validate_text_rows(record.text, record.token_tags)
    values = (record.video, record.audio, record.text, *record.visual_conditions, *record.audio_conditions)
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in values):
        raise ValueError("MiniMaxH3 cache values must be finite FP32")
    if single:
        if item.target_index is None or item.audio_present or item.crop_start_frame or item.condition_roles:
            raise ValueError("MiniMaxH3 single-frame cache requires target_index, silent audio and no video crop/roles")
        overrides = MiniMaxH3TimeOverrides(
            tuple(index * FRAME_RESCALE for index in item.control_indices), item.target_index * FRAME_RESCALE
        )
    else:
        if item.target_index is not None or item.control_indices:
            raise ValueError("MiniMaxH3 temporal indices require single-frame cache")
        overrides = None
    references: list[MiniMaxH3ReferenceGeometry] = []
    visual_geometries = tuple(_video_geometry(value) for value in record.visual_conditions)
    audio_lengths = tuple(_audio_frames(value) for value in record.audio_conditions)
    if format.task == "ref2va":
        if item.condition_roles or item.control_indices:
            raise ValueError("MiniMaxH3 reference caches cannot contain FL conditions")
        kinds = [ref.kind for ref in item.references]
        if kinds.count("image") > 9 or kinds.count("video") > 3:
            raise ValueError("MiniMaxH3 reference cache exceeds9 images or3 videos")
        if sum(ref.kind == "audio" or ref.has_audio for ref in item.references) > 3:
            raise ValueError("MiniMaxH3 reference cache exceeds3 audio-bearing items")
        if single and "audio" in kinds:
            raise ValueError("MiniMaxH3 one-frame targets do not accept standalone audio references")
        visual_count = sum(kind != "audio" for kind in kinds)
        audio_count = sum(ref.kind == "audio" or ref.has_audio for ref in item.references)
        if len(visual_geometries) != visual_count or len(audio_lengths) != audio_count:
            raise ValueError("MiniMaxH3 cached reference modality inventory differs from item")
        visual_index, audio_index = 0, 0
        for ref in item.references:
            geometry, length = None, 0
            if ref.kind != "audio":
                geometry = visual_geometries[visual_index]
                visual_index += 1
            if ref.kind == "audio" or ref.has_audio:
                length = audio_lengths[audio_index]
                audio_index += 1
            references.append(MiniMaxH3ReferenceGeometry(ref.kind, geometry, length))
    else:
        if item.references or audio_lengths:
            raise ValueError("MiniMaxH3 T2VA/FL2VA caches cannot contain reference/audio conditions")
        if format.task == "t2va" and (item.control_indices or visual_geometries or item.condition_roles):
            raise ValueError("MiniMaxH3 T2VA caches cannot contain visual conditions or indices")
        if format.task == "fl2va" and single and len(item.control_indices) != len(visual_geometries):
            raise ValueError("MiniMaxH3 one-frame FL2VA requires an index per visual condition")
    return build_layout(
        task=format.task,
        text_length=record.text.shape[0],
        target_video=video,
        target_audio_frames=audio_frames,
        visual_conditions=visual_geometries if format.task == "fl2va" else (),
        references=references,
        one_frame=single,
        condition_roles=item.condition_roles if format.task == "fl2va" and not single else None,
        time_overrides=overrides,
    )


def _record_tensors(record: MiniMaxH3CacheRecord, format: MiniMaxH3CacheFormat) -> dict[str, Tensor]:
    layout = build_record_layout(record, format=format)
    visual_roles = tuple(segment.role for segment in layout.segments if segment.kind == "visual_condition")
    audio_roles = tuple(segment.role for segment in layout.segments if segment.kind == "audio_condition")
    tensors = {"video": record.video, "audio": record.audio, "text": record.text, "token_tags": record.token_tags}
    tensors.update(
        {f"visual.{role}": value for role, value in zip(visual_roles, record.visual_conditions, strict=True)}
    )
    tensors.update({f"audio.{role}": value for role, value in zip(audio_roles, record.audio_conditions, strict=True)})
    return tensors


def save_minimax_h3_record(
    directory: Path, identity: CacheIdentity, record: MiniMaxH3CacheRecord, *, format: MiniMaxH3CacheFormat
) -> CacheEntry:
    """Atomically publish one complete record, rejecting numerical identity collisions.

    Inputs remain caller-owned; gradients are detached only for serialization.
    Identity includes the corrected audio accounting policy through layout.
    Concurrent writes/mutation to the same identity are unsupported.
    """
    if identity.layout != format.layout:
        raise ValueError("MiniMaxH3 cache identity layout differs from format")
    tensors = _record_tensors(record, format)
    metadata = {
        "unituner.cache.identity": identity.model_dump_json(),
        "unituner.cache.format": format.model_dump_json(),
        "unituner.cache.item": record.item.model_dump_json(),
    }
    return save_cache_record(directory, identity, tensors, metadata=metadata)


def condition_roles(
    item: MiniMaxH3CacheItem, *, format: MiniMaxH3CacheFormat
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ordered visual/audio role inventories declared by one native cache item."""
    if format.task == "ref2va":
        visual = tuple(
            reference_condition_role(i, ref.kind) for i, ref in enumerate(item.references) if ref.kind != "audio"
        )
        audio = tuple(
            reference_condition_role(i, "audio")
            for i, ref in enumerate(item.references)
            if ref.kind == "audio" or ref.has_audio
        )
        return visual, audio
    return (
        one_frame_condition_roles(len(item.control_indices))
        if format.frame_layout == "single-frame"
        else item.condition_roles
    ), ()


def read_minimax_h3_record(
    directory: Path, entry: CacheEntry, *, format: MiniMaxH3CacheFormat, device: torch.device
) -> MiniMaxH3CacheRecord:
    """Verify full file/format/content identity and return owned tensors on the chosen device.

    No implicit dtype conversion, normalization or repacking occurs. Tags remain
    on CPU, matching the text encoder boundary; other values copy to device.
    """
    path = directory / entry.file
    if file_sha256(path) != entry.sha256:
        raise ValueError("MiniMaxH3 cache file differs from manifest digest")
    with TensorReader(path) as reader:
        metadata = reader.metadata()
        if set(metadata) != {"unituner.cache.identity", "unituner.cache.format", "unituner.cache.item"}:
            raise ValueError("MiniMaxH3 cache metadata inventory differs")
        identity = CacheIdentity.model_validate_json(metadata["unituner.cache.identity"])
        stored_format = MiniMaxH3CacheFormat.model_validate_json(metadata["unituner.cache.format"])
        if (
            stored_format != format
            or identity.layout != format.layout
            or identity.digest() + ".safetensors" != entry.file
        ):
            raise ValueError("MiniMaxH3 cache format/identity differs from manifest or filename")
        item = MiniMaxH3CacheItem.model_validate_json(metadata["unituner.cache.item"])
        values = {key: reader.read(key) for key in reader.keys()}
    if not {"video", "audio", "text", "token_tags"} <= values.keys():
        raise ValueError("MiniMaxH3 cache target/text tensors are missing")
    visual_roles, audio_roles = condition_roles(item, format=format)
    expected = {
        "video",
        "audio",
        "text",
        "token_tags",
        *(f"visual.{r}" for r in visual_roles),
        *(f"audio.{r}" for r in audio_roles),
    }
    if values.keys() != expected:
        raise ValueError("MiniMaxH3 cache tensor inventory differs from item")
    record = MiniMaxH3CacheRecord(
        item,
        values["video"],
        values["audio"],
        values["text"],
        values["token_tags"],
        tuple(values[f"visual.{role}"] for role in visual_roles),
        tuple(values[f"audio.{role}"] for role in audio_roles),
    )
    build_record_layout(record, format=format)
    return MiniMaxH3CacheRecord(
        item,
        record.video.to(device, copy=True),
        record.audio.to(device, copy=True),
        record.text.to(device, copy=True),
        record.token_tags.to("cpu", copy=True),
        tuple(value.to(device, copy=True) for value in record.visual_conditions),
        tuple(value.to(device, copy=True) for value in record.audio_conditions),
    )


def save_minimax_h3_manifest(manifest: MiniMaxH3CacheManifest, directory: Path) -> Path:
    """Atomically publish only after every ordered record passes digest and geometry checks."""
    format = MiniMaxH3CacheFormat.model_validate(manifest.model_dump(exclude={"entries"}))
    for entry in manifest.entries:
        read_minimax_h3_record(directory, entry, format=format, device=torch.device("cpu"))
    path = directory / "manifest.json"
    with atomic_output(path) as temporary:
        temporary.write_text(manifest.model_dump_json(indent=2) + "\n")
    return path


def load_minimax_h3_manifest(directory: Path) -> MiniMaxH3CacheManifest:
    """Preflight every complete AV record before training or producer reuse."""
    manifest = MiniMaxH3CacheManifest.model_validate_json((directory / "manifest.json").read_text())
    format = MiniMaxH3CacheFormat.model_validate(manifest.model_dump(exclude={"entries"}))
    for entry in manifest.entries:
        read_minimax_h3_record(directory, entry, format=format, device=torch.device("cpu"))
    return manifest
