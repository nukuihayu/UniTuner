"""Validate AV row boundaries, native modality times and invalid geometry contracts."""

import pytest
import torch

from tests.support.assertions import invalid_input
from unitunner.models.minimax_h3.layout import (
    MiniMaxH3ReferenceGeometry,
    MiniMaxH3TimeOverrides,
    MiniMaxH3VideoGeometry,
    audio_latent_frames,
    build_layout,
    one_frame_condition_role,
    parse_condition_role,
    reference_condition_role,
    video_latent_frames,
)
from unitunner.models.minimax_h3.packing import pack_audio_rows, pack_video_rows, unpack_targets
from unitunner.models.minimax_h3.positions import build_position_grid
from unitunner.models.minimax_h3.timesteps import build_timestep_rows


def test_native_modality_time_rows_and_rotary_anchors():
    layout = build_layout(
        task="fl2va",
        text_length=2,
        target_video=(2, 2, 2),
        target_audio_frames=8,
        visual_conditions=((1, 2, 2), (1, 2, 2)),
    )
    assert [(s.role, s.start, s.stop) for s in layout.segments] == [
        ("text", 0, 2),
        ("first", 2, 3),
        ("last", 3, 4),
        ("target_audio", 4, 20),
        ("target_video", 20, 22),
    ]
    rows = build_timestep_rows(
        layout,
        text_token_tags=torch.tensor([[0, 1]]),
        model_t_video=0.25,
        model_t_audio=0.5,
        visual_condition_clean=0.75,
        audio_condition_clean=1.0,
    )
    assert torch.equal(rows.unique_timesteps, torch.tensor([0.25, 0.5, 0.75]))
    assert torch.equal(rows.row_timesteps, torch.tensor([[0.25, 0.25, 0.75, 0.75] + [0.5] * 16 + [0.25] * 2]))
    assert torch.equal(rows.token_tags, torch.tensor([[0, 1, 0, 0] + [2] * 16 + [0] * 2]))
    assert rows.video_timestep_index == 0 and rows.audio_timestep_index == 1
    positions = build_position_grid(layout)
    assert positions.dtype == torch.float64
    assert positions[0, 2, 0].item() == 2.0
    assert positions[0, 3, 0].item() == 2.0 + 4 * (5 / 3)
    assert torch.equal(positions[0, 4:12, 0], torch.arange(2, 10, dtype=torch.float64))
    assert torch.equal(positions[0, 4:12], positions[0, 12:20])
    assert positions[0, 20, 0].item() == 2.0
    assert positions[0, 21, 0].item() == 2.0 + 5 / 3


def test_invalid_geometry_cannot_truncate_or_break_role_round_trip():
    for geometry in ((2, 4.5, 6), (True, 4, 6), (2, 3, 6), (0, 4, 6)):
        with pytest.raises(ValueError):
            invalid_input(MiniMaxH3VideoGeometry)(*geometry)
    for index in (0, 999):
        assert parse_condition_role(one_frame_condition_role(invalid_input(index))).index == index
        assert parse_condition_role(reference_condition_role(index, "audio")).index == index
    for index in (-1, 1000, 1.5, True):
        with pytest.raises(ValueError):
            one_frame_condition_role(invalid_input(index))
        with pytest.raises(ValueError):
            reference_condition_role(invalid_input(index), "image")
    with pytest.raises(ValueError):
        MiniMaxH3ReferenceGeometry("audio", audio_frames=invalid_input(1.5))
    with pytest.raises(ValueError):
        build_layout(task="t2va", text_length=2, target_video=(7, 4, 6), target_audio_frames=36)
    with pytest.raises(ValueError):
        build_layout(
            task="t2va",
            text_length=2,
            target_video=(7, 4, 6),
            target_audio_frames=37,
            temporal_fine_bands=invalid_input(0.5),
        )
    with pytest.raises(ValueError):
        video_latent_frames(1)
    assert video_latent_frames(124) == 37
    assert audio_latent_frames(124) == 207
    assert audio_latent_frames(124, output_fps=12) == 413


def test_one_frame_restore_rejects_modality_batch_mismatch_and_empty_inputs():
    layout = build_layout(
        task="t2va",
        text_length=2,
        target_video=(1, 2, 2),
        target_audio_frames=2,
        one_frame=True,
        time_overrides=MiniMaxH3TimeOverrides((), 5 / 3),
    )
    video = torch.arange(96).reshape(1, 24, 1, 2, 2)
    audio = torch.arange(128).reshape(1, 32, 2, 2)
    restored = unpack_targets(layout, pack_video_rows(video), pack_audio_rows(audio))
    assert torch.equal(restored.video, video) and torch.equal(restored.audio, audio)
    assert torch.equal(pack_video_rows(video)[0, 0], torch.arange(96))
    assert torch.equal(pack_audio_rows(audio)[0, 0], torch.arange(0, 128, 4))
    with pytest.raises(ValueError):
        unpack_targets(layout, pack_video_rows(video), pack_audio_rows(audio[0]))
    with pytest.raises(ValueError):
        pack_video_rows(torch.empty(1, 24, 0, 2, 2))
    with pytest.raises(ValueError):
        pack_audio_rows(torch.empty(1, 32, 2, 0))
