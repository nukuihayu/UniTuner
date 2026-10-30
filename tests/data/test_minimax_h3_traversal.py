"""Check actual H3 traversal clocks, pixel ownership and strict source choices."""

import json
from fractions import Fraction

import av
import numpy as np
import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from unituner.config.datasets import VideoDatasetConfig
from unituner.data.media.video import select_video_windows
from unituner.models.minimax_h3.config.media import MiniMaxH3RecordInput
from unituner.models.minimax_h3.data.cache.inputs import MiniMaxH3CachePolicy
from unituner.models.minimax_h3.data.cache.storage import MiniMaxH3CacheFormat
from unituner.models.minimax_h3.data.cache.traversal import MiniMaxH3TraversalPolicy, traverse_minimax_h3
from unituner.models.minimax_h3.data.sources import resolve_record


def _video(path, audio_samples=60000):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "bgr0"
        audio = container.add_stream("pcm_f32le", rate=32000)
        audio.layout = "stereo"
        for index in range(44):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), index, np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        waveform = np.stack(
            [np.arange(audio_samples, dtype=np.float32) / 100000, -np.arange(audio_samples, dtype=np.float32) / 100000]
        )
        frame = av.AudioFrame.from_ndarray(waveform, format="fltp", layout="stereo")
        frame.sample_rate, frame.pts, frame.time_base = 32000, 0, Fraction(1, 32000)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    return waveform


def _record(directory, **fields):
    return resolve_record(
        MiniMaxH3RecordInput.model_validate_json(json.dumps({"target_path": "clip.mkv", "caption": "", **fields})),
        directory,
        "t2va",
    )


def _policy(**fields):
    return MiniMaxH3TraversalPolicy(
        resolution=(32, 32),
        video=VideoDatasetConfig(target_frames=(22,), target_fps=24.0, fps_resampling="timestamps", **fields),
    )


def test_real_video_windows_audio_crop_and_owned_storage(tmp_path):
    waveform = torch.from_numpy(_video(tmp_path / "clip.mkv"))
    record = _record(tmp_path)
    rng = torch.get_rng_state().clone()
    items = list(
        traverse_minimax_h3(
            [record],
            format=MiniMaxH3CacheFormat(task="t2va"),
            traversal=_policy(extraction="uniform", samples=3),
            policy=MiniMaxH3CachePolicy(allow_experimental_duration=True),
        )
    )
    assert [item.crop_start_frame for item in items] == [0, 11, 23]
    for item in items:
        start = (item.crop_start_frame * 32000 + 12) // 24
        expected = waveform[:, start : start + 29600]
        expected = torch.nn.functional.pad(expected, (0, 29600 - expected.shape[1]))
        assert torch.equal(item.waveform, expected)
        assert torch.equal(
            item.frames[:, 0, 0, 0],
            torch.arange(item.crop_start_frame, item.crop_start_frame + 22, dtype=torch.uint8).clamp_max(43),
        )
        assert item.audio_source is not None
        assert item.audio_present and item.audio_source.embedded
    items[0].waveform.zero_()
    items[0].frames.zero_()
    assert torch.count_nonzero(items[1].waveform) and items[1].frames[0, 0, 0, 0] == 11
    assert torch.equal(rng, torch.get_rng_state())
    silent = next(
        traverse_minimax_h3(
            [_record(tmp_path, target_audio_path=None)],
            format=MiniMaxH3CacheFormat(task="t2va"),
            traversal=_policy(),
            policy=MiniMaxH3CachePolicy(allow_experimental_duration=True),
        )
    )
    assert not silent.audio_present and not silent.waveform.count_nonzero() and silent.audio_source is None


def test_explicit_container_own_clock_and_short_audio_rejection(tmp_path):
    _video(tmp_path / "clip.mkv")
    record = _record(tmp_path, target_audio_path="clip.mkv")
    item = next(
        traverse_minimax_h3(
            [record],
            format=MiniMaxH3CacheFormat(task="t2va"),
            traversal=_policy(),
            policy=MiniMaxH3CachePolicy(allow_experimental_duration=True),
        )
    )
    assert item.audio_source is not None
    assert item.audio_source.path == record.target_path and not item.audio_source.embedded
    with pytest.raises(ValueError, match="5-15 seconds"):
        list(
            traverse_minimax_h3(
                [record], format=MiniMaxH3CacheFormat(task="t2va"), traversal=_policy(), policy=MiniMaxH3CachePolicy()
            )
        )
    _video(tmp_path / "short.mkv", audio_samples=1000)
    with pytest.raises(ValueError, match="short by"):
        list(
            traverse_minimax_h3(
                [_record(tmp_path, target_audio_path="short.mkv")],
                format=MiniMaxH3CacheFormat(task="t2va"),
                traversal=_policy(),
                policy=MiniMaxH3CachePolicy(allow_experimental_duration=True),
            )
        )


def test_single_frame_rgba_controls_keep_order_and_duplicate_indices(tmp_path):
    pixels = np.arange(32 * 64 * 4, dtype=np.uint8).reshape(32, 64, 4)
    Image.fromarray(pixels).save(tmp_path / "target.png")
    Image.fromarray(pixels[..., :3]).save(tmp_path / "control.png")
    parsed = MiniMaxH3RecordInput.model_validate_json(
        json.dumps(
            {
                "target_path": "target.png",
                "caption": "",
                "target_index": 7,
                "controls": [
                    {"path": "control.png", "index": 9},
                    {"path": "target.png", "index": 2},
                    {"path": "control.png", "index": 9},
                ],
            }
        )
    )
    record = resolve_record(parsed, tmp_path, "fl2va", one_frame=True)
    item = next(
        traverse_minimax_h3(
            [record],
            format=MiniMaxH3CacheFormat(task="fl2va", frame_layout="single-frame"),
            traversal=MiniMaxH3TraversalPolicy(resolution=(32, 32)),
            policy=MiniMaxH3CachePolicy(),
        )
    )
    assert item.target_index == 7 and item.control_indices == (9, 2, 9)
    assert item.frames.shape == (1, 32, 32, 4) and item.control_frames[0].shape == (32, 32, 3)
    assert torch.equal(item.frames[0], torch.from_numpy(pixels[:, 16:48].copy()))
    assert item.waveform.shape == (2, 1600) and not item.waveform.count_nonzero()
    item.control_frames[0].zero_()
    assert item.control_frames[2].count_nonzero()
    with pytest.raises(ValueError, match="one-frame"):
        resolve_record(parsed, tmp_path, "fl2va")
    with pytest.raises(ValueError, match="silent"):
        resolve_record(parsed.model_copy(update={"target_audio_path": "target.png"}), tmp_path, "fl2va", one_frame=True)
    with pytest.raises(ValueError, match="FL2VA"):
        list(
            traverse_minimax_h3(
                [record],
                format=MiniMaxH3CacheFormat(task="t2va", frame_layout="single-frame"),
                traversal=MiniMaxH3TraversalPolicy(resolution=(32, 32)),
                policy=MiniMaxH3CachePolicy(),
            )
        )


@pytest.mark.parametrize(
    "extraction,expected",
    [
        ("head", [(0, 22)]),
        ("chunk", [(0, 22), (22, 22)]),
        ("slide", [(0, 22), (10, 22), (20, 22)]),
        ("uniform", [(0, 22), (11, 22), (22, 22)]),
        ("full", [(0, 39)]),
    ],
)
def test_native_temporal_grid_all_window_modes(extraction, expected):
    video = _policy(extraction=extraction, stride=10, samples=3).video
    assert [
        (item.start, item.count) for item in select_video_windows(44, video, temporal_factor=17, temporal_offset=5)
    ] == expected
    with pytest.raises(ValueError):
        select_video_windows(4, video, temporal_factor=17, temporal_offset=5)


@pytest.mark.parametrize(
    "fields",
    [
        {"resolution": (31, 32)},
        {"resolution": (True, 32)},
        {"video": VideoDatasetConfig(target_frames=(81,), target_fps=24.0, fps_resampling="timestamps")},
        {"video": VideoDatasetConfig(target_frames=(22,), target_fps=24.0)},
    ],
)
def test_invalid_geometry(fields):
    with pytest.raises((ValueError, ValidationError)):
        MiniMaxH3TraversalPolicy(**({"resolution": (32, 32)} | fields))


def test_explicit_video_bounds_keep_absolute_posterior_and_audio_window_coordinates(tmp_path):
    waveform = torch.from_numpy(_video(tmp_path / "clip.mkv"))
    traversal = MiniMaxH3TraversalPolicy(resolution=(32, 32), video=_policy().video, start_frame=7, end_frame=29)
    (item,) = traverse_minimax_h3(
        [_record(tmp_path)],
        format=MiniMaxH3CacheFormat(task="t2va"),
        traversal=traversal,
        policy=MiniMaxH3CachePolicy(allow_experimental_duration=True),
    )
    assert item.crop_start_frame == 7 and len(item.frames) == 22
    assert torch.equal(item.frames[:, 0, 0, 0], torch.arange(7, 29, dtype=torch.uint8))
    assert torch.equal(item.waveform, waveform[:, 9333:38933])
    with pytest.raises(ValueError, match="ordered"):
        MiniMaxH3TraversalPolicy(resolution=(32, 32), start_frame=7, end_frame=7)
