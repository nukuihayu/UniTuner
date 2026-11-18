"""Resolve strict H3 media records and ordered reference policies without decoding pixels."""

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from unituner.data.media.audio import AudioSource
from unituner.data.media.probe import MediaInfo, probe_media
from unituner.models.minimax_h3.config.media import (
    MiniMaxH3AudioReferenceInput,
    MiniMaxH3RecordInput,
    MiniMaxH3ReferenceInput,
    MiniMaxH3VideoReferenceInput,
)
from unituner.models.minimax_h3.layout import MiniMaxH3ReferenceKind, MiniMaxH3Task


@dataclass(frozen=True, slots=True)
class MiniMaxH3Reference:
    """Keep ordered media identity and explicit embedded versus independent audio clocks."""

    kind: MiniMaxH3ReferenceKind
    path: Path
    audio: AudioSource | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("image", "video", "audio"):
            raise ValueError("MiniMaxH3 reference kind must be image/video/audio")
        if self.kind == "image" and (self.audio is not None or self.duration_seconds is not None):
            raise ValueError("MiniMaxH3 image references cannot contain audio or duration")
        if self.kind == "audio" and (self.audio is None or self.audio.embedded or self.audio.path != self.path):
            raise ValueError("MiniMaxH3 standalone audio requires its own independent audio source")
        if self.kind == "video":
            duration = self.duration_seconds
            if duration is None or not math.isfinite(duration) or not 2 <= duration <= 15:
                raise ValueError("MiniMaxH3 reference video duration must be between 2 and 15 seconds")
            if self.audio is not None and self.audio.embedded and self.audio.path != self.path:
                raise ValueError("MiniMaxH3 embedded audio must belong to its reference video")


@dataclass(frozen=True, slots=True)
class MiniMaxH3Control:
    """Keep an ordered source image and nonnegative native24Hz coordinate."""

    path: Path
    index: int

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("MiniMaxH3 control index must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class MiniMaxH3Record:
    """Keep resolved source paths and captions; label is diagnostic, never cache identity."""

    target_path: Path
    caption: str
    references: tuple[MiniMaxH3Reference, ...]
    label: str = ""
    teacher_caption: str | None = None
    target_audio_path: Path | None = None
    target_audio_enabled: bool = True
    target_index: int | None = None
    controls: tuple[MiniMaxH3Control, ...] = ()


def _existing_path(value: str, directory: Path, field: str) -> Path:
    if not value.strip():
        raise ValueError(f"{field} requires a nonempty path")
    path = Path(value)
    path = (directory / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise ValueError(f"{field} does not exist: {path}")
    return path


def validate_reference_policy(
    references: Sequence[MiniMaxH3Reference], task: MiniMaxH3Task, *, one_frame: bool = False
) -> None:
    """Enforce released reference counts and image-target audio restrictions.

    At most twelve items, nine images, three videos and three audio-bearing
    references are allowed, with at least one visual item for Ref2VA. Video
    audio counts toward the same audio limit. Single-frame targets allow video
    audio but cannot define the duration of standalone audio references.
    """
    if task not in ("t2va", "fl2va", "ref2va"):
        raise ValueError(f"Unsupported MiniMaxH3 task: {task}")
    if task != "ref2va":
        if references:
            raise ValueError(f"MiniMaxH3 {task} does not accept references")
        return
    if len(references) > 12:
        raise ValueError("MiniMaxH3 allows at most 12 reference items")
    if any(ref.kind not in ("image", "video", "audio") for ref in references):
        raise ValueError("MiniMaxH3 reference kind must be image/video/audio")
    image_count = sum(ref.kind == "image" for ref in references)
    video_count = sum(ref.kind == "video" for ref in references)
    if image_count > 9:
        raise ValueError("MiniMaxH3 allows at most 9 image references")
    if video_count > 3:
        raise ValueError("MiniMaxH3 allows at most 3 video references")
    if sum(ref.audio is not None for ref in references) > 3:
        raise ValueError("MiniMaxH3 allows at most 3 audio-bearing references")
    if not image_count + video_count:
        raise ValueError("MiniMaxH3 Ref2VA requires at least one visual reference")
    if one_frame and any(ref.kind == "audio" for ref in references):
        raise ValueError("MiniMaxH3 one-frame targets do not accept standalone audio references")


def resolve_references(
    inputs: Sequence[MiniMaxH3ReferenceInput],
    directory: Path,
    task: MiniMaxH3Task,
    *,
    one_frame: bool = False,
    prompt: str | None = None,
    probe: Callable[[Path], MediaInfo] = probe_media,
) -> tuple[MiniMaxH3Reference, ...]:
    """Resolve ordered local references and stream clocks without a target or dummy record.

    Omitted video audio_path selects embedded audio; explicit null suppresses it.
    Explicit paths retain their independent timestamp origins. Validate native
    count/duration policies before any pixel decoding or model construction.
    """
    references: list[MiniMaxH3Reference] = []
    for index, item in enumerate(inputs):
        field = f"references[{index}]"
        path = _existing_path(item.path, directory, f"{field}.path")
        audio, duration = None, None
        if isinstance(item, MiniMaxH3AudioReferenceInput):
            info = probe(path)
            if not info.has_audio:
                raise ValueError(f"{field} contains no audio stream: {path}")
            audio, duration = AudioSource(path, embedded=False), info.duration_seconds
        elif isinstance(item, MiniMaxH3VideoReferenceInput):
            info = probe(path)
            duration = info.duration_seconds
            if not info.has_video:
                raise ValueError(f"{field} contains no video stream: {path}")
            if duration is None or not math.isfinite(duration) or not 2 <= duration <= 15:
                raise ValueError(f"{field} video duration must be between 2 and 15 seconds; got {duration}")
            if not item.audio_enabled:
                audio = None
            elif "audio_path" not in item.model_fields_set:
                if info.has_audio:
                    audio = AudioSource(path, embedded=True)
            elif item.audio_path is not None:
                audio_path = _existing_path(item.audio_path, directory, f"{field}.audio_path")
                if not probe(audio_path).has_audio:
                    raise ValueError(f"{field}.audio_path contains no audio stream: {audio_path}")
                audio = AudioSource(audio_path, embedded=False)
        references.append(MiniMaxH3Reference(item.kind, path, audio, duration))
    validate_reference_policy(references, task, one_frame=one_frame)
    return tuple(references)


def resolve_record(
    record: MiniMaxH3RecordInput,
    directory: Path,
    task: MiniMaxH3Task,
    *,
    one_frame: bool = False,
    label: str = "",
    probe: Callable[[Path], MediaInfo] = probe_media,
) -> MiniMaxH3Record:
    """Resolve existing paths and probe reference streams without reading target pixels.

    Reference videos must have a video stream and finite declared duration of
    2-15 seconds. Embedded audio is considered only for an omitted audio_path;
    an explicit file always has its own clock, even for the same container.
    File/probe errors propagate; JSONL loading adds the manifest line context.
    """
    if task not in ("t2va", "fl2va", "ref2va"):
        raise ValueError(f"Unsupported MiniMaxH3 task: {task}")
    if task != "ref2va" and record.references:
        raise ValueError(f"MiniMaxH3 {task} does not accept references")
    if not one_frame and (record.target_index is not None or record.controls):
        raise ValueError("MiniMaxH3 target_index/controls require one-frame targets")
    if record.controls and task != "fl2va":
        raise ValueError("MiniMaxH3 controls require one-frame FL2VA")
    if one_frame and record.target_audio_path is not None:
        raise ValueError("MiniMaxH3 one-frame targets require silent audio")
    target = _existing_path(record.target_path, directory, "target_path")
    target_audio = None
    if record.target_audio_path is not None:
        target_audio = _existing_path(record.target_audio_path, directory, "target_audio_path")
        if not probe(target_audio).has_audio:
            raise ValueError(f"target_audio_path contains no audio stream: {target_audio}")
    controls = tuple(
        MiniMaxH3Control(_existing_path(item.path, directory, f"controls[{index}].path"), item.index)
        for index, item in enumerate(record.controls)
    )
    references = resolve_references(record.references, directory, task, one_frame=one_frame, probe=probe)
    return MiniMaxH3Record(
        target,
        record.caption,
        tuple(references),
        label,
        record.teacher_caption,
        target_audio,
        "target_audio_path" not in record.model_fields_set or target_audio is not None,
        record.target_index,
        controls,
    )


def load_records(
    path: Path,
    task: MiniMaxH3Task,
    *,
    one_frame: bool = False,
    prompt: str | None = None,
    probe: Callable[[Path], MediaInfo] = probe_media,
) -> tuple[MiniMaxH3Record, ...]:
    """Read nonempty UTF-8 JSONL records relative to the manifest directory.

    Unknown fields and legacy names are rejected. Blank lines are ignored;
    invalid records report the manifest path and physical line number. All
    opened files close on failure. The manifest and media remain unchanged.
    """
    if task not in ("t2va", "fl2va", "ref2va"):
        raise ValueError(f"Unsupported MiniMaxH3 task: {task}")
    path = path.resolve()
    records: list[MiniMaxH3Record] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            label = f"{path} line {line_number}"
            try:
                if prompt is not None:
                    document = json.loads(line)
                    if not isinstance(document, dict):
                        raise ValueError("MiniMaxH3 records must be JSON objects")
                    document["caption"] = prompt
                    line = json.dumps(document)
                data = MiniMaxH3RecordInput.model_validate_json(line)
                records.append(resolve_record(data, path.parent, task, one_frame=one_frame, label=label, probe=probe))
            except (ValueError, OSError) as error:
                detail = (
                    error.errors(include_url=False, include_input=False)
                    if isinstance(error, ValidationError)
                    else str(error)
                )
                raise ValueError(f"MiniMaxH3 {label}: {detail}") from error
    if not records:
        raise ValueError(f"MiniMaxH3 manifest contains no records: {path}")
    return tuple(records)
