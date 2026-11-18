"""Validate H3 manifest boundaries, reference order and explicit audio clock choices."""

import json

import pytest
from pydantic import ValidationError

from tests.support.assertions import invalid_input
from unituner.data.media.audio import AudioSource
from unituner.data.media.probe import MediaInfo
from unituner.models.minimax_h3.config.media import MiniMaxH3RecordInput
from unituner.models.minimax_h3.data.sources import (
    MiniMaxH3Reference,
    load_records,
    resolve_record,
    validate_reference_policy,
)


@pytest.fixture
def media(tmp_path):
    for name in ("target.png", "image.png", "video.mkv", "audio.wav"):
        (tmp_path / name).touch()
    return tmp_path


def _record(references):
    return MiniMaxH3RecordInput.model_validate_json(
        json.dumps({"target_path": "target.png", "caption": "", "references": references})
    )


def _probe(path):
    return MediaInfo(path.suffix == ".mkv", path.suffix != ".png", 3.0)


def test_order_omitted_null_explicit_and_same_container_audio_clocks(media):
    items = [
        {"kind": "video", "path": "video.mkv"},
        {"kind": "image", "path": "image.png"},
        {"kind": "video", "path": "video.mkv", "audio_path": None},
        {"kind": "video", "path": "video.mkv", "audio_path": "video.mkv"},
        {"kind": "audio", "path": "audio.wav"},
    ]
    parsed = _record(items)
    record = resolve_record(parsed, media, "ref2va", probe=_probe)
    assert [ref.kind for ref in record.references] == [item["kind"] for item in items]
    assert [None if r.audio is None else r.audio.embedded for r in record.references] == [
        True,
        None,
        None,
        False,
        False,
    ]
    assert record.target_path == media / "target.png" and record.caption == ""
    assert parsed == MiniMaxH3RecordInput.model_validate_json(parsed.model_dump_json(exclude_unset=True))
    with pytest.raises(ValueError, match="standalone audio"):
        resolve_record(parsed, media, "ref2va", one_frame=True, probe=_probe)
    resolve_record(_record(items[:-1]), media, "ref2va", one_frame=True, probe=_probe)


def test_explicit_video_audio_disable_matches_json_null(media):
    records = [
        resolve_record(_record([{"kind": "video", "path": "video.mkv", **fields}]), media, "ref2va", probe=_probe)
        for fields in ({}, {"audio_path": None}, {"audio_enabled": False})
    ]
    assert records[0].references[0].audio == AudioSource(media / "video.mkv", True)
    assert records[1].references == records[2].references
    assert records[2].references[0].audio is None
    for fields in ({"audio_enabled": False, "audio_path": "missing.wav"}, {"audio_enabled": 0}):
        with pytest.raises(ValidationError):
            _record([{"kind": "video", "path": "video.mkv", **fields}])


@pytest.mark.parametrize("task", ["t2va", "fl2va"])
def test_non_reference_tasks_and_invalid_task_reject_before_probe(media, task):
    resolve_record(_record([]), media, task, probe=_probe)
    with pytest.raises(ValueError, match="does not accept references"):
        resolve_record(_record([{"kind": "video", "path": "missing"}]), media, task, probe=_probe)
    with pytest.raises(ValueError, match="Unsupported"):
        load_records(media / "missing.jsonl", invalid_input("unknown"))


@pytest.mark.parametrize("duration", [None, float("nan"), float("inf"), 1.999, 15.001])
def test_invalid_video_duration(media, duration):
    with pytest.raises(ValueError, match="between 2 and 15"):
        resolve_record(
            _record([{"kind": "video", "path": "video.mkv"}]),
            media,
            "ref2va",
            probe=lambda p: MediaInfo(True, False, duration),
        )


@pytest.mark.parametrize("duration", [2.0, 15.0])
def test_inclusive_video_duration_boundaries(media, duration):
    record = resolve_record(
        _record([{"kind": "video", "path": "video.mkv"}]),
        media,
        "ref2va",
        probe=lambda p: MediaInfo(True, False, duration),
    )
    assert record.references[0].duration_seconds == duration


def test_missing_streams_and_explicit_file(media):
    with pytest.raises(ValueError, match="no video stream"):
        resolve_record(_record([{"kind": "video", "path": "audio.wav"}]), media, "ref2va", probe=_probe)
    for items in (
        [{"kind": "image", "path": "image.png"}, {"kind": "audio", "path": "image.png"}],
        [{"kind": "video", "path": "video.mkv", "audio_path": "image.png"}],
    ):
        with pytest.raises(ValueError, match="no audio stream"):
            resolve_record(_record(items), media, "ref2va", probe=_probe)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_record(_record([{"kind": "image", "path": "missing.png"}]), media, "ref2va", probe=_probe)


@pytest.mark.parametrize(
    "fields",
    [
        {"video_path": "target.png", "caption": ""},
        {"target_path": "target.png", "caption": 1},
        {"target_path": "target.png", "caption": "", "teacher_caption": "  "},
        {"target_path": "target.png", "caption": "", "rcm_mask": True},
        {"target_path": "target.png", "caption": "", "references": [{"type": "image", "path": "image.png"}]},
        {
            "target_path": "target.png",
            "caption": "",
            "references": [{"kind": "image", "path": "image.png", "audio_path": None}],
        },
        {
            "target_path": "target.png",
            "caption": "",
            "references": [{"kind": "audio", "path": "audio.wav", "audio_path": None}],
        },
    ],
)
def test_strict_native_schema(fields):
    with pytest.raises(ValidationError):
        MiniMaxH3RecordInput.model_validate_json(json.dumps(fields))


def test_released_reference_limits_and_audio_bearing_total(media):
    image = MiniMaxH3Reference("image", media / "image.png")
    audio = MiniMaxH3Reference("audio", media / "audio.wav", AudioSource(media / "audio.wav", False), 3.0)
    video = MiniMaxH3Reference("video", media / "video.mkv", AudioSource(media / "video.mkv", True), 3.0)
    validate_reference_policy([image] * 9 + [video] * 3, "ref2va")
    validate_reference_policy([image] * 9 + [audio] * 3, "ref2va")
    for refs, message in (
        ([], "visual"),
        ([audio], "visual"),
        ([image] * 10, "9 image"),
        ([video] * 4, "3 video"),
        ([video] * 3 + [audio], "3 audio-bearing"),
        ([image] * 9 + [video] * 3 + [audio], "12 reference"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_reference_policy(refs, "ref2va")


def test_manifest_relative_paths_blank_lines_and_line_errors(media, monkeypatch):
    path = media / "records.jsonl"
    content = "\n" + json.dumps({"target_path": "target.png", "caption": "雪", "teacher_caption": "主体"}) + "\n"
    path.write_text(content, encoding="utf-8")
    monkeypatch.chdir(media.parent)
    (record,) = load_records(path, "t2va")
    assert record.target_path == media / "target.png" and record.caption == "雪"
    assert record.teacher_caption == "主体" and record.label.endswith("line 2")
    assert path.read_text(encoding="utf-8") == content
    for invalid in ('{"caption": 4}', "[]", "{invalid", '{"target_path": "missing", "caption": ""}'):
        path.write_text(content + invalid, encoding="utf-8")
        with pytest.raises(ValueError, match=r"records.jsonl line 3"):
            load_records(path, "t2va")
    path.write_text(" \n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        load_records(path, "t2va")


@pytest.mark.parametrize("prompt", ["shared caption", ""])
def test_global_prompt_overrides_or_replaces_missing_record_captions(media, prompt):
    path = media / "records.jsonl"
    path.write_text(json.dumps({"target_path": "target.png"}) + "\n")
    with pytest.raises(ValueError, match="caption"):
        load_records(path, "t2va")
    assert load_records(path, "t2va", prompt=prompt)[0].caption == prompt
    path.write_text(json.dumps({"target_path": "target.png", "caption": "ignored"}) + "\n")
    assert load_records(path, "t2va", prompt=prompt)[0].caption == prompt


def test_reference_record_invariants(media):
    for args in (
        ("image", media / "image.png", AudioSource(media / "audio.wav", False)),
        ("audio", media / "audio.wav"),
        ("audio", media / "audio.wav", AudioSource(media / "audio.wav", True)),
        ("video", media / "video.mkv", None, float("nan")),
        ("video", media / "video.mkv", AudioSource(media / "audio.wav", True), 3.0),
    ):
        with pytest.raises(ValueError):
            invalid_input(MiniMaxH3Reference)(*args)
