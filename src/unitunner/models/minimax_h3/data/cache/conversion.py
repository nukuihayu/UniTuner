"""Convert explicit original split H3 caches after revalidating source media clocks and geometry."""

import json
import re
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from unitunner.artifacts.hashing import canonical_sha256, file_sha256
from unitunner.artifacts.tensors import TensorReader
from unitunner.config.datasets import VideoDatasetConfig
from unitunner.data.cache.schema import CacheIdentity
from unitunner.data.cache.storage import CacheEntry
from unitunner.data.media.audio import decode_audio, resolve_audio_source
from unitunner.data.media.audio_timeline import slice_audio_window
from unitunner.models.minimax_h3.config.media import MiniMaxH3RecordInput
from unitunner.models.minimax_h3.data.cache.inputs import MiniMaxH3CachePolicy, MiniMaxH3DecodedItem, build_cache_item
from unitunner.models.minimax_h3.data.cache.storage import (
    MiniMaxH3CacheFormat,
    MiniMaxH3CacheItem,
    MiniMaxH3CacheManifest,
    MiniMaxH3CacheRecord,
    build_record_layout,
    condition_roles,
    load_minimax_h3_manifest,
    save_minimax_h3_manifest,
    save_minimax_h3_record,
)
from unitunner.models.minimax_h3.data.cache.traversal import MiniMaxH3TraversalPolicy, traverse_minimax_h3
from unitunner.models.minimax_h3.data.media.inputs import read_reference_image, read_reference_video
from unitunner.models.minimax_h3.data.sources import MiniMaxH3Record, resolve_record
from unitunner.models.minimax_h3.layout import audio_latent_frames, video_latent_frames

_FLOAT_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class MiniMaxH3ImportEntry(BaseModel):
    """Pin both input cache files, native source declaration and complete target layout."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    latents: str = Field(min_length=1)
    text: str = Field(min_length=1)
    latents_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: MiniMaxH3RecordInput
    item: MiniMaxH3CacheItem
    resolution: tuple[int, int]


class MiniMaxH3ImportSpec(BaseModel):
    """Declare an offline split-to-native conversion, never a runtime loading fallback.

    Paths are relative to this JSON specification. encoder_sha256 declares the
    original encoder artifact identity; this converter does not load encoders
    or prove that cached numbers were computed by those weights. Exact floating
    promotion is explicit and does not resample or renormalize any latent.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal[1] = 1
    source_format: Literal["minimax-h3-split/1"]
    float_conversion: Literal["exact-fp32"]
    encoder_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: MiniMaxH3CacheFormat
    policy: MiniMaxH3CachePolicy = Field(default_factory=MiniMaxH3CachePolicy)
    entries: tuple[MiniMaxH3ImportEntry, ...] = Field(min_length=1)


def _read_values(path: Path) -> dict[str, Tensor]:
    with TensorReader(path) as reader:
        return {key: reader.read(key) for key in reader.keys()}


def _pop_float(values: dict[str, Tensor], prefix: str, *, video: bool = False, shape_suffix: bool = True) -> Tensor:
    """Require one exact external key, matching dtype/shape suffix and finite values."""
    shape = r"([1-9][0-9]*(?:x[1-9][0-9]*){2})_" if shape_suffix else ""
    pattern = re.compile(re.escape(prefix) + shape + r"(float32|float16|bfloat16)")
    matches = [(key, match) for key in values if (match := pattern.fullmatch(key)) is not None]
    if len(matches) != 1:
        raise ValueError(f"MiniMaxH3 split cache requires exactly one {prefix} tensor")
    key, match = matches[0]
    value = values.pop(key)
    if value.dtype != _FLOAT_DTYPES[match.groups()[-1]] or not torch.isfinite(value).all():
        raise ValueError(f"MiniMaxH3 split cache dtype/finite mismatch: {key}")
    if shape_suffix:
        declared = tuple(int(axis) for axis in match.group(1).split("x"))
        if declared != (tuple(value.shape[1:]) if video else tuple(value.shape)):
            raise ValueError(f"MiniMaxH3 split cache shape suffix differs: {key}")
    return value.float().clone()


def _pop_integer(values: dict[str, Tensor], key: str, expected: Tensor) -> None:
    value = values.pop(key, None)
    if value is None or value.dtype != torch.int64 or value.shape != expected.shape or not torch.equal(value, expected):
        raise ValueError(f"MiniMaxH3 split cache coordinate differs: {key}")


def read_split_record(
    latents: Path, text: Path, item: MiniMaxH3CacheItem, *, format: MiniMaxH3CacheFormat
) -> MiniMaxH3CacheRecord:
    """Read owned exact FP32 AV/text values with strict original keys and no extra tensors.

    Recognize FP32/FP16/BF16 suffixes and require actual dtypes to match. Audio
    presence and temporal coordinates must equal the explicit native item.
    Teacher, RCM and all unknown tensor fields reject rather than being dropped.
    Input metadata is not copied; complete file digests retain its provenance.
    """
    values, features = _read_values(latents), _read_values(text)
    presence = values.pop("audio_present_float32", None)
    if (
        presence is None
        or presence.dtype != torch.float32
        or presence.shape != ()
        or presence.item() != float(item.audio_present)
    ):
        raise ValueError("MiniMaxH3 split cache audio presence differs from declared item")
    if item.target_index is not None:
        _pop_integer(values, "one_frame_target_index_int64", torch.tensor(item.target_index, dtype=torch.int64))
    if item.control_indices:
        _pop_integer(values, "one_frame_control_indices_int64", torch.tensor(item.control_indices, dtype=torch.int64))
    visual_roles, audio_roles = condition_roles(item, format=format)
    video = _pop_float(values, "latents_", video=True)
    audio = _pop_float(values, "latents_audio_")
    visuals = tuple(_pop_float(values, f"latents_{role}_", video=True) for role in visual_roles)
    audios = tuple(_pop_float(values, f"latents_{role}_") for role in audio_roles)
    hidden = _pop_float(features, "varlen_mmh3_hidden_states_", shape_suffix=False)
    tags = features.pop("varlen_mmh3_token_tags_int64", None)
    if tags is None:
        raise ValueError("MiniMaxH3 split cache student token tags are missing")
    if values or features:
        raise ValueError("MiniMaxH3 split cache has unknown, teacher or RCM tensor fields")
    record = MiniMaxH3CacheRecord(item, video, audio, hidden, tags.clone(), visuals, audios)
    build_record_layout(record, format=format)
    return record


def _source_paths(record: MiniMaxH3Record, *, single: bool) -> set[Path]:
    paths = {record.target_path, *(control.path for control in record.controls)}
    for reference in record.references:
        paths.add(reference.path)
        if reference.audio is not None:
            paths.add(reference.audio.path)
    if not single and record.target_audio_enabled:
        audio = resolve_audio_source(record.target_path, explicit_path=record.target_audio_path)
        if audio is not None:
            paths.add(audio.path)
    return paths


def _validate_media(
    record: MiniMaxH3CacheRecord,
    decoded: MiniMaxH3DecodedItem,
    *,
    format: MiniMaxH3CacheFormat,
    policy: MiniMaxH3CachePolicy,
) -> None:
    """Validate real source clocks/repair budgets and VAE grids without running an encoder."""
    expected = build_cache_item(decoded, format, policy)
    if format.task == "fl2va" and format.frame_layout == "video":
        if record.item.condition_roles not in (("first",), ("last",), ("first", "last")):
            raise ValueError("MiniMaxH3 split FL video requires first/last condition roles")
        expected = expected.model_copy(update={"condition_roles": record.item.condition_roles})
    if record.item != expected:
        raise ValueError("MiniMaxH3 split item differs from actual source window/conditions")
    height, width = decoded.frames.shape[1:3]
    if tuple(record.video.shape[-2:]) != (height // 16, width // 16):
        raise ValueError("MiniMaxH3 split target latent canvas differs from source resolution")
    single = format.frame_layout == "single-frame"
    if format.task != "ref2va":
        if any(tuple(value.shape) != (24, 1, height // 16, width // 16) for value in record.visual_conditions):
            raise ValueError("MiniMaxH3 split FL condition grid differs from target canvas")
        return
    visual_index, audio_index = 0, 0
    for reference in decoded.record.references:
        frames = None
        if reference.kind == "image":
            frames = read_reference_image(reference.path, target_size=(width, height))
        elif reference.kind == "video":
            frames = read_reference_video(reference.path, target_frame_count=360 if single else len(decoded.frames))
        if frames is not None:
            count = 1 if reference.kind == "image" else video_latent_frames(len(frames))
            expected_shape = (24, count, frames.shape[1] // 16, frames.shape[2] // 16)
            if tuple(record.visual_conditions[visual_index].shape) != expected_shape:
                raise ValueError("MiniMaxH3 split reference visual grid differs from source media")
            visual_index += 1
        if reference.audio is not None:
            samples = audio_latent_frames(len(frames)) * 800 if frames is not None else decoded.waveform.shape[1]
            audio = decode_audio(reference.audio, policy.audio_decode)
            waveform = slice_audio_window(
                audio.waveform,
                start_sample=0,
                sample_count=samples,
                require_exact=reference.kind == "video",
                repairs=audio.repairs,
                pad_tolerance=policy.audio_pad_tolerance,
                max_repair_samples=policy.audio_max_repair_samples,
                context=str(reference.path),
            )
            if record.audio_conditions[audio_index].shape[-1] != (waveform.shape[1] + 799) // 800:
                raise ValueError("MiniMaxH3 split reference audio grid differs from source window")
            audio_index += 1


def import_minimax_h3_cache(source: Path, destination: Path) -> MiniMaxH3CacheManifest:
    """Publish a native cache from a strict JSON conversion specification.

    Re-read original target/reference media, including timestamp alignment and
    corrected window-local repair budgets; source media must remain available.
    Preserve all cached numeric values with explicit exact FP32 promotion.
    Verify declared input hashes before IO and source hashes across conversion.
    Publish immutable records then the manifest atomically; existing records and
    manifest survive failure. Conversion has a distinct transform identity and
    never claims to have recomputed VAE/text features or verified encoder weights.
    """
    source, destination = source.resolve(), destination.resolve()
    spec_digest = file_sha256(source)
    spec = MiniMaxH3ImportSpec.model_validate_json(source.read_text())
    single = spec.format.frame_layout == "single-frame"
    if (destination / "manifest.json").exists():
        previous = load_minimax_h3_manifest(destination)
        if MiniMaxH3CacheFormat.model_validate(previous.model_dump(exclude={"entries"})) != spec.format:
            raise ValueError("MiniMaxH3 import destination has a different cache format")
    entries: list[CacheEntry] = []
    all_files: dict[Path, str] = {source: spec_digest}
    for entry in spec.entries:
        latents, text = (source.parent / entry.latents).resolve(), (source.parent / entry.text).resolve()
        declared = {latents: entry.latents_sha256, text: entry.text_sha256}
        if latents == text:
            raise ValueError("MiniMaxH3 split latent/text files must be distinct")
        if any(file_sha256(path) != digest for path, digest in declared.items()):
            raise ValueError("MiniMaxH3 split input file differs from declared digest")
        original = resolve_record(entry.source, source.parent, spec.format.task, one_frame=single)
        if original.teacher_caption is not None:
            raise ValueError("MiniMaxH3 teacher cache import is not implemented")
        paths = _source_paths(original, single=single)
        files = {**declared, **{path: file_sha256(path) for path in paths}}
        if any(path in all_files and all_files[path] != digest for path, digest in files.items()):
            raise ValueError("MiniMaxH3 import source changed across entries")
        all_files.update(files)
        if destination / "manifest.json" in all_files:
            raise ValueError("MiniMaxH3 output manifest must not overwrite conversion inputs")
        record = read_split_record(latents, text, entry.item, format=spec.format)
        traversal = (
            MiniMaxH3TraversalPolicy(resolution=entry.resolution)
            if single
            else MiniMaxH3TraversalPolicy(
                resolution=entry.resolution,
                start_frame=entry.item.crop_start_frame,
                end_frame=entry.item.crop_start_frame + entry.item.frame_count,
                video=VideoDatasetConfig(
                    target_frames=(entry.item.frame_count,), target_fps=24.0, fps_resampling="timestamps"
                ),
            )
        )
        (decoded,) = traverse_minimax_h3([original], format=spec.format, traversal=traversal, policy=spec.policy)
        _validate_media(record, decoded, format=spec.format, policy=spec.policy)
        if paths != _source_paths(original, single=single) or any(
            file_sha256(path) != digest for path, digest in files.items()
        ):
            raise ValueError("MiniMaxH3 source changed during cache conversion")
        identity = CacheIdentity(
            source_sha256=canonical_sha256(
                {
                    "files": {str(path): digest for path, digest in sorted(files.items())},
                    "source": entry.source.model_dump(mode="json", exclude_unset=True),
                }
            ),
            encoder_sha256=spec.encoder_sha256,
            transform_sha256=canonical_sha256(
                {
                    "schema": "minimax-h3/imported-split-cache/1",
                    "float_conversion": spec.float_conversion,
                    "item": entry.item.model_dump(mode="json"),
                    "resolution": list(entry.resolution),
                    "policy": spec.policy.model_dump(mode="json"),
                    "format": spec.format.model_dump(mode="json"),
                }
            ),
            layout=spec.format.layout,
        )
        if destination / (identity.digest() + ".safetensors") in all_files:
            raise ValueError("MiniMaxH3 output record must not overwrite conversion inputs")
        entries.append(save_minimax_h3_record(destination, identity, record, format=spec.format))
    if any(file_sha256(path) != digest for path, digest in all_files.items()):
        raise ValueError("MiniMaxH3 import inputs changed before manifest publication")
    manifest = MiniMaxH3CacheManifest(**spec.format.model_dump(), entries=tuple(entries))
    save_minimax_h3_manifest(manifest, destination)
    return manifest


def run(specification: Path, destination: Path) -> int:
    """Validate source media and publish complete native AV/student-text cache records."""
    manifest = import_minimax_h3_cache(specification, destination)
    print(json.dumps({"converted": len(manifest.entries), "family": manifest.family}))
    return 0
