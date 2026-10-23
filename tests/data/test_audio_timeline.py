"""Validate audio alignment, differentiable repairs and window-local silence budgets."""

import pytest
import torch
from pydantic import ValidationError

from unitunner.config.audio import AudioDecodeConfig
from unitunner.data.media.audio_timeline import (
    AudioRepair,
    assemble_audio_chunks,
    audio_window_start,
    repaired_samples_in_window,
    slice_audio_window,
)


def test_gap_overlap_origin_and_floating_gradients():
    first = torch.arange(20, dtype=torch.float64).reshape(2, 10).requires_grad_()
    second = torch.arange(8, dtype=torch.float64).reshape(2, 4).requires_grad_()
    third = torch.arange(16, dtype=torch.float64).reshape(2, 8).requires_grad_()
    chunks = [(0, first), (13, second), (15, third)]
    decoded = assemble_audio_chunks(
        chunks, channels=2, timestamp_tolerance_samples=1, max_gap_fill_samples=3, max_overlap_trim_samples=2
    )
    expected = torch.cat((first, torch.zeros(2, 3), second, third[:, 2:]), dim=1)
    assert torch.equal(decoded.waveform, expected)
    assert decoded.repairs == (AudioRepair(10, filled=3), AudioRepair(17, trimmed=2))
    before = assemble_audio_chunks(
        chunks,
        channels=2,
        timestamp_tolerance_samples=1,
        max_gap_fill_samples=3,
        max_overlap_trim_samples=2,
        origin_sample=12,
    )
    assert torch.equal(before.waveform, expected[:, 12:])
    assert before.repairs == (AudioRepair(0, filled=1), AudioRepair(5, trimmed=2))
    after = assemble_audio_chunks(
        chunks,
        channels=2,
        timestamp_tolerance_samples=1,
        max_gap_fill_samples=3,
        max_overlap_trim_samples=2,
        origin_sample=-4,
    )
    assert torch.equal(after.waveform, torch.cat((torch.zeros(2, 4), expected), dim=1))
    assert after.repairs == (AudioRepair(0, filled=4), AudioRepair(14, filled=3), AudioRepair(21, trimmed=2))
    decoded.waveform.sum().backward()
    assert first.grad is not None and second.grad is not None
    assert torch.equal(first.grad, torch.ones_like(first)) and torch.equal(second.grad, torch.ones_like(second))
    assert third.grad is not None
    assert torch.equal(third.grad, torch.cat((torch.zeros(2, 2), torch.ones(2, 6)), dim=1))


def test_bidirectional_jitter_is_distinct_from_one_way_drift():
    data = [torch.full((2, 10), float(i)) for i in range(3)]
    jitter = assemble_audio_chunks(list(zip((0, 13, 20), data, strict=True)), channels=2, pts_jitter_range_samples=3)
    assert torch.equal(jitter.waveform, torch.cat(data, dim=1)) and not jitter.repairs
    drift = assemble_audio_chunks(
        list(zip((0, 13, 26), data, strict=True)), channels=2, pts_jitter_range_samples=6, max_gap_fill_samples=3
    )
    assert drift.repairs == (AudioRepair(10, filled=3), AudioRepair(23, filled=3))
    with pytest.raises(ValueError, match="discontinuity"):
        assemble_audio_chunks([(0, data[0]), (14, data[1])], channels=2, max_gap_fill_samples=3)


def test_window_budget_charges_silence_intersections_and_native_terminal_padding():
    repairs = (AudioRepair(10, filled=8), AudioRepair(20, trimmed=3))
    assert repaired_samples_in_window(repairs, 13, 2) == 2
    assert repaired_samples_in_window(repairs, 8, 4) == 2
    assert repaired_samples_in_window(repairs, 18, 2) == 0
    assert repaired_samples_in_window(repairs, 18, 3) == 3
    waveform = torch.ones(2, 30)
    with pytest.raises(ValueError, match="budget"):
        slice_audio_window(waveform, start_sample=13, sample_count=2, repairs=repairs, max_repair_samples=1)
    assert slice_audio_window(
        waveform, start_sample=8, sample_count=4, repairs=repairs, max_repair_samples=2
    ).shape == (2, 4)
    padded = slice_audio_window(waveform, start_sample=28, sample_count=5, pad_tolerance=2, max_repair_samples=3)
    assert torch.equal(padded, torch.tensor([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float32))
    with pytest.raises(ValueError, match="materially short"):
        slice_audio_window(waveform, start_sample=28, sample_count=5, pad_tolerance=2, max_repair_samples=2)
    assert slice_audio_window(waveform, start_sample=28, sample_count=5, require_exact=False).shape == (2, 2)
    assert audio_window_start(1, 24, 32000) == 1333 and audio_window_start(2, 24, 32000) == 2667


def test_invalid_audio_config_chunks_and_repair_geometry():
    for fields in (
        {"sample_rate": 0, "channels": 2},
        {"sample_rate": 32000, "channels": 3},
        {"sample_rate": 32000, "channels": True},
        {"sample_rate": 32000, "channels": 2, "max_discontinuity_seconds": float("nan")},
    ):
        with pytest.raises(ValidationError):
            AudioDecodeConfig(**fields)
    with pytest.raises(ValueError, match="position"):
        AudioRepair(-1, filled=2)
    with pytest.raises(ValueError, match="nonempty"):
        assemble_audio_chunks([(0, torch.empty(2, 0))], channels=2)
    with pytest.raises(ValueError, match="finite"):
        assemble_audio_chunks([(0, torch.full((2, 2), float("nan")))], channels=2)
    with pytest.raises(ValueError, match="common floating"):
        assemble_audio_chunks([(0, torch.ones(2, 2)), (2, torch.ones(2, 2, dtype=torch.float64))], channels=2)
