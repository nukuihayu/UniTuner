# Copyright 2025-2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Diffusers PR #14355 (abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc)
# via the pinned Musubi and old MixTuner implementations.
"""Describe native AV row geometry without media IO or training dependencies."""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

TARGET_FPS = 24
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
STEREO_CHANNELS = 2
VIDEO_PATCH_SIZE = (1, 2, 2)
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
ONE_FRAME_VIDEO_LATENT_FRAMES = 1
ONE_FRAME_AUDIO_LATENT_FRAMES = 2
MiniMaxH3Task = Literal["t2va", "fl2va", "ref2va"]
MiniMaxH3ReferenceKind = Literal["image", "audio", "video"]
MiniMaxH3SegmentKind = Literal["text", "visual_condition", "audio_condition", "target_audio", "target_video"]
MiniMaxH3ConditionFamily = Literal["fl", "one_frame", "reference"]
FL_CONDITION_ROLES = ("first", "last")
REFERENCE_KINDS = ("image", "video", "audio")
_ONE_FRAME_CONDITION_ROLE = re.compile(r"^cond_(\d{3})$")
_REFERENCE_CONDITION_ROLE = re.compile(r"^ref_(\d{3})_(image|video|audio)$")


def video_latent_frames(frame_count: int) -> int:
    """Map native17*n+5 decoded frames to5*n+2 video latent frames."""
    if type(frame_count) is not int or frame_count < 5 or (frame_count - 5) % 17:
        raise ValueError("MiniMaxH3 video frame_count must be an integer17*n+5")
    return 5 * ((frame_count - 5) // 17) + 2


def audio_latent_frames(frame_count: int, *, output_fps: int = TARGET_FPS) -> int:
    """Map a native video frame count to40Hz audio latent frames at explicit FPS."""
    video_latent_frames(frame_count)
    if type(output_fps) is not int or output_fps <= 0:
        raise ValueError("MiniMaxH3 output_fps must be a positive integer")
    if output_fps == TARGET_FPS:
        return (10 * frame_count + 3) // 6
    return int((Fraction(10 * frame_count * TARGET_FPS, output_fps) + 3) // 6)


@dataclass(frozen=True, slots=True)
class MiniMaxH3VideoGeometry:
    """Describe positive latent FHW geometry with spatial2x2 patch alignment."""

    frames: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.frames, self.height, self.width)):
            raise ValueError("MiniMaxH3 video geometry requires integer axes")
        if self.frames <= 0:
            raise ValueError(f"MiniMax-H3 video geometry needs positive frames, got {self.frames}")
        if self.height <= 0 or self.width <= 0 or self.height % 2 or self.width % 2:
            raise ValueError(
                f"MiniMax-H3 latent video axes must be positive multiples of 2, got {self.height}x{self.width}"
            )

    @property
    def frame_rows(self) -> int:
        return (self.height // 2) * (self.width // 2)

    @property
    def row_count(self) -> int:
        return self.frames * self.frame_rows


@dataclass(frozen=True, slots=True)
class MiniMaxH3ReferenceGeometry:
    """Describe one ordered visual/audio reference and its synchronized row counts."""

    kind: MiniMaxH3ReferenceKind
    video: MiniMaxH3VideoGeometry | None = None
    audio_frames: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("image", "audio", "video"):
            raise ValueError(f"Unsupported MiniMax-H3 reference kind: {self.kind}")
        if type(self.audio_frames) is not int or self.audio_frames < 0:
            raise ValueError(f"MiniMax-H3 reference audio length must be nonnegative, got {self.audio_frames}")
        if self.kind == "image":
            if self.video is None or self.video.frames != 1 or self.audio_frames:
                raise ValueError("MiniMax-H3 image references require one visual frame and no audio rows")
        elif self.kind == "audio":
            if self.video is not None or self.audio_frames <= 0:
                raise ValueError("MiniMax-H3 audio references require audio rows and no visual geometry")
        else:
            if self.video is None:
                raise ValueError("MiniMax-H3 video references require visual geometry")
            _validate_video_latent_frames(self.video.frames, "reference video")
            if self.audio_frames:
                expected = _expected_audio_frames(self.video.frames)
                if self.audio_frames != expected:
                    raise ValueError(
                        f"MiniMax-H3 reference video audio has {self.audio_frames} frames, expected {expected}"
                    )

    @property
    def visual_geometry(self) -> MiniMaxH3VideoGeometry:
        """Return the geometry of a visual reference; audio-only references fail."""
        if self.video is None:
            raise ValueError("An audio reference has no visual geometry")
        return self.video


def one_frame_condition_role(index: int) -> str:
    """Role of the index-th one-frame FL2VA visual condition (``cond_000``, ``cond_001``, ...).

    One-frame conditions are placed by explicit time overrides, so unlike the released video
    FL2VA roles (``first``/``last``, whose names select the anchor time) they are plain ordered
    slots within the three-digit role vocabulary; text numbers ``<Picture i>`` in the same order.
    """
    if type(index) is not int or not 0 <= index <= 999:
        raise ValueError(f"MiniMaxH3 one-frame condition index must be an integer in[0,999], got {index}")
    return f"cond_{index:03d}"


def one_frame_condition_roles(count: int) -> tuple[str, ...]:
    """Return ordered condition roles within the fixed three-digit artifact vocabulary."""
    if type(count) is not int or not 0 <= count <= 1000:
        raise ValueError("MiniMaxH3 condition count must be an integer in[0,1000]")
    return tuple(one_frame_condition_role(index) for index in range(count))


def reference_condition_role(index: int, kind: MiniMaxH3ReferenceKind) -> str:
    """Role of the index-th Ref2VA reference's ``kind`` rows (``ref_000_image``, ``ref_001_audio``, ...).

    A video reference with audio owns two roles (``_video`` and ``_audio``) under one index.
    """
    if type(index) is not int or not 0 <= index <= 999:
        raise ValueError(f"MiniMaxH3 reference index must be an integer in[0,999], got {index}")
    if kind not in REFERENCE_KINDS:
        raise ValueError(f"Unsupported MiniMax-H3 reference kind: {kind}")
    return f"ref_{index:03d}_{kind}"


@dataclass(frozen=True, slots=True)
class MiniMaxH3ConditionRole:
    """A parsed condition role name (see the vocabulary above).

    ``index`` is the one-frame slot or reference number (None for the ``first``/``last``
    anchors); ``reference_kind`` is the reference row type (None outside the reference family).
    """

    name: str
    family: MiniMaxH3ConditionFamily
    index: int | None = None
    reference_kind: MiniMaxH3ReferenceKind | None = None

    @property
    def is_audio(self) -> bool:
        return self.reference_kind == "audio"


def parse_condition_role(role: str) -> MiniMaxH3ConditionRole:
    """Parses a condition role name; raises ValueError for anything outside the vocabulary."""
    if role in FL_CONDITION_ROLES:
        return MiniMaxH3ConditionRole(role, "fl")
    match = _ONE_FRAME_CONDITION_ROLE.fullmatch(role)
    if match is not None:
        return MiniMaxH3ConditionRole(role, "one_frame", index=int(match.group(1)))
    match = _REFERENCE_CONDITION_ROLE.fullmatch(role)
    if match is not None:
        kind = match.group(2)
        if kind not in ("image", "audio", "video"):
            raise ValueError(f"Unsupported MiniMax-H3 reference kind: {kind}")
        return MiniMaxH3ConditionRole(role, "reference", index=int(match.group(1)), reference_kind=kind)
    raise ValueError(f"Unsupported MiniMax-H3 condition role: {role}")


@dataclass(frozen=True, slots=True)
class MiniMaxH3TimeOverrides:
    """Explicit RoPE times for a one-frame layout, in rotary units (1 unit = 1/40 s).

    Times are relative to the target-block cursor (text end for fl2va, after the
    reference blocks for ref2va); only relative placement carries meaning. A 24 fps
    pixel-frame index maps to FRAME_RESCALE * index units.
    """

    condition_times: tuple[float, ...]
    target_time: float

    def __post_init__(self) -> None:
        for label, value in (
            *((f"condition time {i}", t) for i, t in enumerate(self.condition_times)),
            ("target time", self.target_time),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"MiniMax-H3 {label} must be finite and nonnegative, got {value}")


@dataclass(frozen=True, slots=True)
class MiniMaxH3RowSegment:
    """Describe one nonempty half-open segment of the shared packed sequence."""

    role: str
    kind: MiniMaxH3SegmentKind
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(f"Invalid MiniMax-H3 row segment {self.role}: [{self.start}, {self.stop})")

    @property
    def row_count(self) -> int:
        return self.stop - self.start

    @property
    def row_slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True, slots=True)
class MiniMaxH3PackedLayout:
    """Own immutable segment order, target geometry and optional temporal overrides."""

    task: MiniMaxH3Task
    text_length: int
    target_video: MiniMaxH3VideoGeometry
    target_audio_frames: int
    visual_conditions: tuple[MiniMaxH3VideoGeometry, ...]
    references: tuple[MiniMaxH3ReferenceGeometry, ...]
    segments: tuple[MiniMaxH3RowSegment, ...]
    row_count: int
    # part of the frozen layout value so the transformer's layout-keyed rotary cache
    # cannot serve a grid built for different times
    time_overrides: MiniMaxH3TimeOverrides | None = None
    # experimental target-timeline stretch: generated pixel frames sample the timeline at
    # output_fps instead of the native 24 fps, so each frame spans 24/output_fps native
    # rotary steps and the clip covers frame_count/output_fps real seconds. References and
    # conditions keep their native 24 fps spans; the target audio frame count must cover
    # the stretched real duration. Frozen here so the rotary cache keys on it.
    output_fps: int = TARGET_FPS
    # with a stretch, rotate this many leading (highest-frequency) temporal RoPE bands
    # by the UNSTRETCHED grid instead. Those bands have periods at or below the latent
    # token spacing, so they cannot encode time between tokens -- they carry a per-token
    # lattice phase that training bakes in against the native (1,4,4,4,4)-span grid;
    # stretching scrambles it with the 17-pixel-frame group period (fading/stripes).
    # ~3 is the sub-lattice band count for the released 16-band spectrum.
    temporal_fine_bands: int = 0

    def segment(self, role: str) -> MiniMaxH3RowSegment:
        matches = [segment for segment in self.segments if segment.role == role]
        if len(matches) != 1:
            raise KeyError(f"MiniMax-H3 layout expected exactly one {role!r} segment, found {len(matches)}")
        return matches[0]

    @property
    def temporal_stretch(self) -> float:
        return TARGET_FPS / self.output_fps

    @property
    def target_video_segment(self) -> MiniMaxH3RowSegment:
        return self.segment("target_video")

    @property
    def target_audio_segment(self) -> MiniMaxH3RowSegment:
        return self.segment("target_audio")


def _coerce_video_geometry(value: MiniMaxH3VideoGeometry | Sequence[int], label: str) -> MiniMaxH3VideoGeometry:
    """Resolve validated FHW geometry without truncating fractional dimensions."""
    if isinstance(value, MiniMaxH3VideoGeometry):
        return value
    if len(value) != 3:
        raise ValueError(f"MiniMax-H3 {label} geometry must be (frames,height,width), got {tuple(value)}")
    return MiniMaxH3VideoGeometry(*(part for part in value))


def _validate_video_latent_frames(frames: int, label: str) -> None:
    """Require the native video latent temporal grid5*n+2."""
    if type(frames) is not int or frames < 2 or (frames - 2) % 5:
        raise ValueError(f"MiniMax-H3 {label} latent frames must be 5*n+2, got {frames}")


def _expected_audio_frames(video_latent_frames: int, output_fps: int = TARGET_FPS) -> int:
    """Compute synchronized audio length from the latent video temporal grid."""
    _validate_video_latent_frames(video_latent_frames, "target video")
    n = (video_latent_frames - 2) // 5
    pixel_frames = 17 * n + 5
    return audio_latent_frames(pixel_frames, output_fps=output_fps)


def build_layout(
    *,
    task: MiniMaxH3Task,
    text_length: int,
    target_video: MiniMaxH3VideoGeometry | Sequence[int],
    target_audio_frames: int,
    visual_conditions: Sequence[MiniMaxH3VideoGeometry | Sequence[int]] = (),
    references: Sequence[MiniMaxH3ReferenceGeometry] = (),
    one_frame: bool = False,
    condition_roles: Sequence[str] | None = None,
    time_overrides: MiniMaxH3TimeOverrides | None = None,
    output_fps: int = TARGET_FPS,
    temporal_fine_bands: int = 0,
) -> MiniMaxH3PackedLayout:
    """Validate native T2VA/FL2VA/Ref2VA geometry and build complete ordered row segments."""
    if task not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError(f"Unsupported MiniMax-H3 task: {task}")
    if type(text_length) is not int or text_length <= 0:
        raise ValueError(f"MiniMax-H3 text length must be positive, got {text_length}")
    if type(target_audio_frames) is not int or target_audio_frames <= 0:
        raise ValueError("MiniMaxH3 target_audio_frames must be a positive integer")
    if isinstance(output_fps, bool) or not isinstance(output_fps, int) or output_fps <= 0:
        raise ValueError(f"MiniMax-H3 output fps must be a positive integer, got {output_fps!r}")
    if one_frame and output_fps != TARGET_FPS:
        raise ValueError("MiniMax-H3 one-frame layouts do not support temporal stretch")
    if type(temporal_fine_bands) is not int:
        raise ValueError("MiniMaxH3 temporal_fine_bands must be an integer")
    if temporal_fine_bands < 0:
        raise ValueError(f"MiniMax-H3 temporal fine bands must be nonnegative, got {temporal_fine_bands}")
    if temporal_fine_bands and output_fps == TARGET_FPS:
        raise ValueError("MiniMax-H3 temporal fine bands require an active temporal stretch")
    target_video = _coerce_video_geometry(target_video, "target video")
    if one_frame:
        if target_video.frames != ONE_FRAME_VIDEO_LATENT_FRAMES:
            raise ValueError(
                f"MiniMax-H3 one-frame layout requires a single target latent frame, got {target_video.frames}"
            )
        if target_audio_frames != ONE_FRAME_AUDIO_LATENT_FRAMES:
            raise ValueError(
                f"MiniMax-H3 one-frame layout requires {ONE_FRAME_AUDIO_LATENT_FRAMES} target audio frames, "
                f"got {target_audio_frames}"
            )
    else:
        if time_overrides is not None:
            raise ValueError("MiniMax-H3 time overrides require a one-frame layout")
        _validate_video_latent_frames(target_video.frames, "target video")
        expected_audio_frames = _expected_audio_frames(target_video.frames, output_fps)
        if target_audio_frames != expected_audio_frames:
            raise ValueError(
                f"MiniMax-H3 target audio has {target_audio_frames} frames, expected {expected_audio_frames} "
                f"for {target_video.frames} video latent frames at {output_fps} fps"
            )

    roles: tuple[str, ...] = ()
    visual_conditions = tuple(
        _coerce_video_geometry(condition, f"visual condition {index}")
        for index, condition in enumerate(visual_conditions)
    )
    references = tuple(references)
    if task == "t2va" and (visual_conditions or references):
        raise ValueError("MiniMax-H3 T2VA layout does not accept condition rows")
    if task == "fl2va":
        if references:
            raise ValueError("MiniMax-H3 FL2VA layout requires exactly first and last visual conditions")
        if one_frame:
            # ordered cond_{i} slots, any count, each placed by its own time override
            if not visual_conditions:
                raise ValueError("MiniMax-H3 one-frame FL2VA layout requires at least one visual condition")
            if time_overrides is None or len(time_overrides.condition_times) != len(visual_conditions):
                raise ValueError("MiniMax-H3 one-frame FL2VA layout requires one condition time override per condition")
            roles = one_frame_condition_roles(len(visual_conditions))
            if condition_roles is not None and tuple(condition_roles) != roles:
                raise ValueError(
                    f"MiniMax-H3 one-frame FL2VA condition roles are the ordered {roles}, got {tuple(condition_roles)}"
                )
        else:
            if not 1 <= len(visual_conditions) <= 2:
                raise ValueError("MiniMax-H3 FL2VA layout requires one or two visual conditions (first and/or last)")
            roles = _fl_condition_roles(condition_roles, len(visual_conditions))
        for role, condition in zip(roles, visual_conditions, strict=True):
            if condition != MiniMaxH3VideoGeometry(1, target_video.height, target_video.width):
                raise ValueError(
                    f"MiniMax-H3 FL2VA {role} condition must be one target-sized latent frame, got {condition}"
                )
    else:
        if condition_roles is not None:
            raise ValueError("MiniMax-H3 condition roles apply only to FL2VA layouts")
    if task == "ref2va":
        if visual_conditions or not references:
            raise ValueError("MiniMax-H3 Ref2VA layout requires ordered references and no FL2VA conditions")
        if not any(reference.kind in {"image", "video"} for reference in references):
            raise ValueError("MiniMax-H3 Ref2VA layout requires at least one visual reference")
    if time_overrides is not None and task != "fl2va" and time_overrides.condition_times:
        raise ValueError(f"MiniMax-H3 {task} time overrides cannot carry condition times")

    segments = []
    row = 0

    def append(role: str, kind: MiniMaxH3SegmentKind, row_count: int) -> None:
        nonlocal row
        segments.append(MiniMaxH3RowSegment(role, kind, row, row + row_count))
        row += row_count

    append("text", "text", text_length)
    if task == "fl2va":
        for role, condition in zip(roles, visual_conditions, strict=True):
            append(role, "visual_condition", condition.row_count)
    elif task == "ref2va":
        for index, reference in enumerate(references):
            if reference.kind == "image":
                append(
                    reference_condition_role(index, "image"), "visual_condition", reference.visual_geometry.row_count
                )
            elif reference.kind == "audio":
                append(
                    reference_condition_role(index, "audio"),
                    "audio_condition",
                    reference.audio_frames * STEREO_CHANNELS,
                )
            else:
                if reference.audio_frames:
                    append(
                        reference_condition_role(index, "audio"),
                        "audio_condition",
                        reference.audio_frames * STEREO_CHANNELS,
                    )
                append(
                    reference_condition_role(index, "video"), "visual_condition", reference.visual_geometry.row_count
                )
    append("target_audio", "target_audio", target_audio_frames * STEREO_CHANNELS)
    append("target_video", "target_video", target_video.row_count)

    return MiniMaxH3PackedLayout(
        task=task,
        text_length=text_length,
        target_video=target_video,
        target_audio_frames=target_audio_frames,
        visual_conditions=visual_conditions,
        references=references,
        segments=tuple(segments),
        row_count=row,
        time_overrides=time_overrides,
        output_fps=output_fps,
        temporal_fine_bands=temporal_fine_bands,
    )


def _fl_condition_roles(condition_roles: Sequence[str] | None, condition_count: int) -> tuple[str, ...]:
    """Resolve validated ordered first/last-frame anchor roles."""
    if condition_roles is None:
        if condition_count != 2:
            raise ValueError("MiniMax-H3 FL2VA layout with a single condition requires explicit condition roles")
        return FL_CONDITION_ROLES
    roles = tuple(condition_roles)
    if roles not in {("first",), ("last",), FL_CONDITION_ROLES}:
        raise ValueError(f"MiniMax-H3 FL2VA condition roles must be first, last, or first+last in order, got {roles}")
    if len(roles) != condition_count:
        raise ValueError(f"MiniMax-H3 FL2VA layout has {condition_count} conditions for {len(roles)} roles")
    return roles
