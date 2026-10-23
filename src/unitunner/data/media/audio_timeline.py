"""Assemble audio sample timelines and enforce window-local missing-sample budgets."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor

_LOGGER = logging.getLogger(__name__)


def _nonnegative_integer(value: int, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class AudioRepair:
    """Locate filled silence and/or a trimmed-overlap event in assembled sample units.

    Filled samples occupy [position,position+filled). Trimmed samples are absent
    from the assembled waveform and are charged at the event's position.
    """

    position: int
    filled: int = 0
    trimmed: int = 0

    def __post_init__(self) -> None:
        for field, value in (("position", self.position), ("filled", self.filled), ("trimmed", self.trimmed)):
            _nonnegative_integer(value, field)


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    """Carry a floating channels-by-samples waveform and repairs on its sample grid."""

    waveform: Tensor
    repairs: tuple[AudioRepair, ...] = ()

    def repaired_samples(self, start_sample: int, sample_count: int) -> int:
        """Count repaired silence intersecting the window and trimmed events within it."""
        return repaired_samples_in_window(self.repairs, start_sample, sample_count)


def repaired_samples_in_window(repairs: Sequence[AudioRepair], start_sample: int, sample_count: int) -> int:
    """Count filled interval intersections and trimmed events on a half-open window.

    A fill beginning before the window still contributes its intersecting samples;
    a fill ending after the window contributes only the part within the window.
    Overlap trims retain the native event-based charge because removed samples
    have no extent on the assembled waveform's timeline.
    """
    _nonnegative_integer(start_sample, "start_sample")
    _nonnegative_integer(sample_count, "sample_count")
    end = start_sample + sample_count
    return sum(
        max(0, min(end, repair.position + repair.filled) - max(start_sample, repair.position))
        + (repair.trimmed if start_sample <= repair.position < end else 0)
        for repair in repairs
    )


def _place_at_origin(decoded: DecodedAudio, offset: int, tolerance: int, context: str) -> DecodedAudio:
    """Shift a waveform origin, retaining repairs only on the remaining sample grid."""
    if offset == 0:
        return decoded
    waveform = decoded.waveform
    if offset > 0:
        silence = waveform.new_zeros(waveform.shape[0], offset)
        repairs = [AudioRepair(0, filled=offset)] if offset > tolerance else []
        repairs.extend(AudioRepair(r.position + offset, r.filled, r.trimmed) for r in decoded.repairs)
        return DecodedAudio(torch.cat((silence, waveform), dim=1), tuple(repairs))
    dropped = -offset
    if dropped >= waveform.shape[1]:
        raise ValueError(f"Audio stream ends before the video origin: {context}")
    repairs = []
    for repair in decoded.repairs:
        position = repair.position - dropped
        if position >= 0:
            repairs.append(AudioRepair(position, repair.filled, repair.trimmed))
        elif repair.filled > -position:
            repairs.append(AudioRepair(0, filled=repair.filled + position))
    return DecodedAudio(waveform[:, dropped:], tuple(repairs))


def assemble_audio_chunks(
    chunks: Sequence[tuple[int, Tensor]],
    *,
    channels: int,
    timestamp_tolerance_samples: int = 2,
    pts_jitter_range_samples: int = 0,
    max_gap_fill_samples: int = 0,
    max_overlap_trim_samples: int = 0,
    origin_sample: int | None = None,
    context: str = "",
) -> DecodedAudio:
    """Place timestamped floating CL chunks without mutating their values or gradients.

    Bounded deviations with both positive and negative steps beyond tolerance are
    timestamp jitter: concatenate in decode order. Otherwise place each chunk at
    its timestamp, filling gaps or trimming overlaps under their explicit limits.
    Discontinuities within tolerance are repaired but unrecorded. Larger repairs
    are recorded for window checks. Sample0 is the first chunk's start unless an
    explicit origin is given; leading silence is recorded and pre-origin audio
    is dropped. Inputs must have common floating dtype/device and nonempty axes.
    """
    if type(channels) is not int or channels not in (1, 2) or not chunks:
        raise ValueError("Audio assembly requires nonempty mono/stereo chunks")
    for field, value in (
        ("timestamp_tolerance_samples", timestamp_tolerance_samples),
        ("pts_jitter_range_samples", pts_jitter_range_samples),
        ("max_gap_fill_samples", max_gap_fill_samples),
        ("max_overlap_trim_samples", max_overlap_trim_samples),
    ):
        _nonnegative_integer(value, field)
    if origin_sample is not None and type(origin_sample) is not int:
        raise ValueError("Audio origin_sample must be an integer or None")
    first = chunks[0][1]
    for start, chunk in chunks:
        if type(start) is not int:
            raise ValueError("Audio chunk timestamps must be integer sample indices")
        if chunk.ndim != 2 or chunk.shape[0] != channels or chunk.shape[1] <= 0:
            raise ValueError("Audio chunks require nonempty channels-by-samples geometry")
        if not chunk.is_floating_point() or chunk.dtype != first.dtype or chunk.device != first.device:
            raise ValueError("Audio chunks require a common floating dtype and device")
        if not torch.isfinite(chunk).all():
            raise ValueError("Audio chunks must contain finite samples")
    offset = 0 if origin_sample is None else chunks[0][0] - origin_sample
    if pts_jitter_range_samples > 0:
        nominal = chunks[0][0]
        deviations = []
        for start, chunk in chunks:
            deviations.append(start - nominal)
            nominal += chunk.shape[1]
        steps = [b - a for a, b in pairwise(deviations)]
        if (
            max(deviations) - min(deviations) <= pts_jitter_range_samples
            and any(step > timestamp_tolerance_samples for step in steps)
            and any(step < -timestamp_tolerance_samples for step in steps)
        ):
            _LOGGER.debug("Audio timestamp jitter spans %d samples", max(deviations) - min(deviations))
            return _place_at_origin(
                DecodedAudio(torch.cat([chunk for _, chunk in chunks], dim=1)),
                offset,
                timestamp_tolerance_samples,
                context,
            )
    expected_start = chunks[0][0]
    position = 0
    assembled: list[Tensor] = []
    repairs: list[AudioRepair] = []
    for start, chunk in chunks:
        delta = start - expected_start
        if abs(delta) > timestamp_tolerance_samples:
            limit = max_gap_fill_samples if delta > 0 else max_overlap_trim_samples
            if abs(delta) > limit:
                raise ValueError(
                    f"Audio discontinuity at sample {position}: jump {delta}, repair limit {limit}: {context}"
                )
            repairs.append(
                AudioRepair(position, filled=delta)
                if delta > 0
                else AudioRepair(position, trimmed=min(-delta, chunk.shape[1]))
            )
        if delta > 0:
            assembled.append(chunk.new_zeros(channels, delta))
            position += delta
        elif delta < 0:
            chunk = chunk[:, -delta:]
        if chunk.shape[1] > 0:
            assembled.append(chunk)
            position += chunk.shape[1]
        expected_start = start + chunk.shape[1] + max(0, -delta)
    return _place_at_origin(
        DecodedAudio(torch.cat(assembled, dim=1), tuple(repairs)), offset, timestamp_tolerance_samples, context
    )


def slice_audio_window(
    waveform: Tensor,
    *,
    start_sample: int,
    sample_count: int,
    pad_tolerance: int = 800,
    require_exact: bool = True,
    repairs: Sequence[AudioRepair] = (),
    max_repair_samples: int | None = None,
    context: str = "",
) -> Tensor:
    """Slice floating CL audio, enforcing repair budgets and native terminal padding.

    Filled silence is charged by its intersection with the requested half-open
    window, with trimmed overlaps charged at their event positions. With exact
    length requested, pad a terminal shortfall up to pad_tolerance uncharged;
    larger shortfalls require enough remaining repair budget. Non-exact windows
    retain a short tail and reject an empty result. Return contiguous storage,
    which can share the input when already contiguous; inputs are not modified
    and floating gradients are retained.
    """
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2) or not waveform.is_floating_point():
        raise ValueError("Audio window requires a floating mono/stereo CL waveform")
    _nonnegative_integer(start_sample, "start_sample")
    _nonnegative_integer(pad_tolerance, "pad_tolerance")
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("Audio window sample_count must be a positive integer")
    if type(require_exact) is not bool:
        raise ValueError("Audio window require_exact must be boolean")
    if max_repair_samples is not None:
        _nonnegative_integer(max_repair_samples, "max_repair_samples")
    repaired = repaired_samples_in_window(repairs, start_sample, sample_count)
    if max_repair_samples is not None and repaired > max_repair_samples:
        raise ValueError(f"Audio window has {repaired} repaired samples, budget {max_repair_samples}: {context}")
    window = waveform[:, start_sample : start_sample + sample_count]
    if require_exact and window.shape[1] < sample_count:
        deficit = sample_count - window.shape[1]
        if deficit > pad_tolerance:
            if max_repair_samples is None or repaired + deficit > max_repair_samples:
                raise ValueError(f"Audio window is materially short by {deficit} samples: {context}")
            _LOGGER.warning("Audio window at sample %d ends %d samples early", start_sample, deficit)
        window = torch.nn.functional.pad(window, (0, deficit))
    if window.shape[1] == 0:
        raise ValueError(f"Audio window is empty at sample {start_sample}: {context}")
    return window.contiguous()


def audio_window_start(crop_start_frame: int, fps: int, sample_rate: int) -> int:
    """Map a nonnegative video frame index to the native nearest audio sample."""
    _nonnegative_integer(crop_start_frame, "crop_start_frame")
    if any(type(value) is not int or value <= 0 for value in (fps, sample_rate)):
        raise ValueError("Audio window fps and sample_rate must be positive integers")
    return (crop_start_frame * sample_rate + fps // 2) // fps
