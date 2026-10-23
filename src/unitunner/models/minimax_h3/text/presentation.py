# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Diffusers PR #14355 (abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc).
"""Build non-chat H3 presentations and native text/vision row labels without media IO."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from unitunner.models.minimax_h3.layout import (
    FL_CONDITION_ROLES,
    MiniMaxH3ReferenceKind,
    MiniMaxH3Task,
    one_frame_condition_roles,
    parse_condition_role,
)

TEXT_VISUAL_FPS = 2
TEXT_WIDTH = 5120
MAX_TEXT_ROWS = 32768
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


@dataclass(frozen=True, slots=True)
class MiniMaxH3TextVisual:
    """Borrow nonempty THWC RGB/RGBA pixels and optional ordered timestamps in seconds.

    Pixels retain their dtype/device and are never modified. Pixel range and
    conversion belong to the processor; timestamps must be finite and nonnegative.
    """

    frames: Tensor
    timestamps: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.frames.ndim != 4 or min(self.frames.shape) <= 0 or self.frames.shape[-1] not in (3, 4):
            raise ValueError("MiniMaxH3 text visuals require nonempty THWC RGB/RGBA pixels")
        if self.timestamps is not None:
            if len(self.timestamps) != len(self.frames):
                raise ValueError("MiniMaxH3 timestamps must match the frame count")
            if any(not math.isfinite(t) or t < 0 for t in self.timestamps) or any(
                a > b for a, b in zip(self.timestamps, self.timestamps[1:], strict=False)
            ):
                raise ValueError("MiniMaxH3 timestamps must be finite, nonnegative and ordered")


@dataclass(frozen=True, slots=True)
class MiniMaxH3TextReference:
    """Describe an ordered reference; audio contributes a label, never vision pixels."""

    kind: MiniMaxH3ReferenceKind
    visual: MiniMaxH3TextVisual | None = None
    has_audio: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("image", "video", "audio"):
            raise ValueError("MiniMaxH3 reference kind must be image/video/audio")
        if self.kind == "audio":
            if self.visual is not None or self.has_audio:
                raise ValueError("MiniMaxH3 standalone audio references cannot contain visual/audio-track fields")
        elif self.visual is None:
            raise ValueError("MiniMaxH3 image/video references require visual pixels")
        elif self.kind == "image" and (len(self.visual.frames) != 1 or self.has_audio):
            raise ValueError("MiniMaxH3 image references require one frame and no audio track")


@dataclass(frozen=True, slots=True)
class MiniMaxH3Presentation:
    """Borrow ordered visual pixels and keep descriptive versus processor video text.

    Descriptive text expands one timestamp per temporal pair. Processor text keeps
    one placeholder per original video, leaving temporal expansion to Qwen3-VL.
    Neither string includes a system prompt or chat wrapper.
    """

    text: str
    processor_text: str
    images: tuple[Tensor, ...] = ()
    videos: tuple[Tensor, ...] = ()


def build_presentation(
    caption: str,
    task: MiniMaxH3Task,
    *,
    anchors: Mapping[str, MiniMaxH3TextVisual] | None = None,
    references: Sequence[MiniMaxH3TextReference] = (),
) -> MiniMaxH3Presentation:
    """Build native T2VA/FL2VA/Ref2VA labels in packed order without RNG or copies.

    FL2VA accepts first/last anchors or contiguous cond_000 slots, never both.
    A last-only anchor is Picture1. Ref2VA counts each media kind independently;
    synchronized video audio precedes its video label. Odd videos repeat only the
    final timestamp for descriptive pairing; processor pixels stay untouched.
    """
    if task not in ("t2va", "fl2va", "ref2va"):
        raise ValueError("MiniMaxH3 task must be t2va/fl2va/ref2va")
    if not isinstance(caption, str):
        raise ValueError("MiniMaxH3 caption must be a string")
    anchors = {} if anchors is None else anchors
    if task != "ref2va" and references:
        raise ValueError(f"MiniMaxH3 {task} does not accept references")
    if task != "fl2va" and anchors:
        raise ValueError(f"MiniMaxH3 {task} does not accept anchors")
    parts: list[str] = []
    processor_parts: list[str] = []
    images: list[Tensor] = []
    videos: list[Tensor] = []
    if task == "fl2va":
        families = {parse_condition_role(role).family for role in anchors}
        if families == {"fl"}:
            roles = tuple(role for role in FL_CONDITION_ROLES if role in anchors)
        elif families == {"one_frame"}:
            roles = tuple(sorted(anchors))
            if roles != one_frame_condition_roles(len(roles)):
                raise ValueError("MiniMaxH3 one-frame anchors require contiguous cond_000 slots")
        else:
            raise ValueError("MiniMaxH3 FL2VA requires first/last or cond_ anchors without mixing")
        for index, role in enumerate(roles, 1):
            visual = anchors[role]
            if len(visual.frames) != 1:
                raise ValueError("MiniMaxH3 FL2VA anchors require exactly one frame")
            parts.append(f"<Picture {index}>: {IMAGE_PLACEHOLDER}")
            images.append(visual.frames[0])
        processor_parts.extend(parts)
    if task == "ref2va":
        counters = {"image": 0, "audio": 0, "video": 0}
        for reference in references:
            if reference.kind == "audio" or reference.has_audio:
                counters["audio"] += 1
                label = f"<Audio {counters['audio']}>: "
                parts.append(label)
                processor_parts.append(label)
            if reference.kind == "audio":
                continue
            visual = reference.visual
            if visual is None:
                raise ValueError("MiniMaxH3 visual reference requires pixels")
            counters[reference.kind] += 1
            if reference.kind == "image":
                label = f"<Picture {counters['image']}>: {IMAGE_PLACEHOLDER}"
                parts.append(label)
                processor_parts.append(label)
                images.append(visual.frames[0])
            else:
                label = f"<Video {counters['video']}>: "
                parts.append(label)
                processor_parts.extend((label, VIDEO_PLACEHOLDER))
                times = visual.timestamps
                if times is None:
                    times = tuple(i / TEXT_VISUAL_FPS for i in range(len(visual.frames)))
                for i in range(0, len(times), 2):
                    midpoint = (times[i] + times[min(i + 1, len(times) - 1)]) / 2.0
                    parts.append(f"<{midpoint:.1f} seconds>{VIDEO_PLACEHOLDER}")
                videos.append(visual.frames)
    parts.append(caption)
    processor_parts.append(caption)
    return MiniMaxH3Presentation("".join(parts), "".join(processor_parts), tuple(images), tuple(videos))


def build_token_tags(input_ids: Tensor) -> Tensor:
    """Return CPU int64 row labels:1 for text and0 for inclusive vision spans.

    IDs must be a nonempty integer L or1xL tensor. Nested/unmatched delimiters
    and spans without image/video rows fail. These are modality labels for native
    AV packing, not an attention mask or a training loss mask.
    """
    if input_ids.ndim == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids[0]
    if input_ids.ndim != 1 or not input_ids.numel() or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("MiniMaxH3 input_ids require nonempty integer L or1xL")
    ids = input_ids.detach().cpu().tolist()
    tags = torch.ones(len(ids), dtype=torch.int64)
    start: int | None = None
    has_visual = False
    for index, token in enumerate(ids):
        if token == VISION_START_TOKEN_ID:
            if start is not None:
                raise ValueError("MiniMaxH3 nested vision-start token")
            start, has_visual = index, False
        elif token == VISION_END_TOKEN_ID:
            if start is None or not has_visual:
                raise ValueError("MiniMaxH3 unmatched or empty vision span")
            tags[start : index + 1] = 0
            start = None
        elif token in (IMAGE_TOKEN_ID, VIDEO_TOKEN_ID) and start is not None:
            has_visual = True
    if start is not None:
        raise ValueError("MiniMaxH3 unmatched vision-start token")
    return tags


def validate_text_rows(hidden_states: Tensor, token_tags: Tensor) -> None:
    """Validate floating Lx5120 features and binary int64 L modality labels.

    Keep the original32768-row cap before downstream AV packing. This boundary
    validates geometry and labels without copying or altering feature tensors.
    """
    if hidden_states.ndim != 2 or hidden_states.shape[1] != TEXT_WIDTH or not hidden_states.is_floating_point():
        raise ValueError("MiniMaxH3 text features require floating Lx5120")
    if not 0 < hidden_states.shape[0] <= MAX_TEXT_ROWS:
        raise ValueError("MiniMaxH3 text rows must be in[1,32768]")
    if token_tags.shape != (hidden_states.shape[0],) or token_tags.dtype != torch.int64:
        raise ValueError("MiniMaxH3 token tags require int64 L")
    if not torch.all((token_tags == 0) | (token_tags == 1)):
        raise ValueError("MiniMaxH3 token tags must be binary")
