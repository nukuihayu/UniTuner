"""Exercise native complete AV cache integrity, timeline and modality geometry."""

from dataclasses import replace

import pytest
import torch
from pydantic import ValidationError

from unitunner.artifacts.hashing import file_sha256
from unitunner.artifacts.tensors import TensorReader, save_tensors
from unitunner.data.cache.schema import CacheIdentity
from unitunner.models.minimax_h3.data.cache.storage import (
    MiniMaxH3CacheFormat,
    MiniMaxH3CacheItem,
    MiniMaxH3CacheManifest,
    MiniMaxH3CacheRecord,
    MiniMaxH3CacheReference,
    build_record_layout,
    load_minimax_h3_manifest,
    read_minimax_h3_record,
    save_minimax_h3_manifest,
    save_minimax_h3_record,
)


def _values(shape, offset=0):
    return (torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape) + offset) / 37


def _fixture(task, single):
    format = MiniMaxH3CacheFormat(task=task, frame_layout="single-frame" if single else "video")
    fields = {"frame_count": 1 if single else 22, "audio_present": not single}
    if single:
        fields["target_index"] = 7
    visual, audio = (), ()
    if task == "fl2va":
        if single:
            fields["control_indices"] = (9, 2, 9)
            visual = tuple(_values((24, 1, 4, 6), offset=i) for i in range(3))
        else:
            fields["condition_roles"] = ("first", "last")
            visual = (_values((24, 1, 4, 6)), _values((24, 1, 4, 6), offset=1))
    elif task == "ref2va":
        refs = (MiniMaxH3CacheReference(kind="image"), MiniMaxH3CacheReference(kind="video", has_audio=True))
        visual = (_values((24, 1, 2, 4)), _values((24, 2, 4, 4)))
        audio = (_values((32, 2, 8)),)
        if not single:
            refs += (MiniMaxH3CacheReference(kind="audio"),)
            audio += (_values((32, 2, 17)),)
        fields["references"] = refs
    record = MiniMaxH3CacheRecord(
        MiniMaxH3CacheItem(**fields),
        _values((24, 1 if single else 7, 4, 6)),
        _values((32, 2, 2 if single else 37)),
        _values((3, 5120)),
        torch.tensor([1, 0, 1], dtype=torch.int64),
        visual,
        audio,
    )
    identity = CacheIdentity(
        source_sha256="1" * 64, encoder_sha256="2" * 64, transform_sha256="3" * 64, layout=format.layout
    )
    return format, record, identity


def _assert_equal(actual, expected):
    assert actual.item == expected.item
    for key in ("video", "audio", "text", "token_tags"):
        value, other = getattr(actual, key), getattr(expected, key)
        assert value.dtype == other.dtype and torch.equal(value, other)
        assert value.data_ptr() != other.data_ptr()
    for key in ("visual_conditions", "audio_conditions"):
        for value, other in zip(getattr(actual, key), getattr(expected, key), strict=True):
            assert torch.equal(value, other) and value.data_ptr() != other.data_ptr()


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va"])
@pytest.mark.parametrize("single", [False, True])
def test_complete_records_roundtrip_and_temporal_layout(tmp_path, task, single):
    format, record, identity = _fixture(task, single)
    rng = torch.get_rng_state().clone()
    entry = save_minimax_h3_record(tmp_path, identity, record, format=format)
    original = (tmp_path / entry.file).read_bytes()
    assert save_minimax_h3_record(tmp_path, identity, record, format=format) == entry
    restored = read_minimax_h3_record(tmp_path, entry, format=format, device=torch.device("cpu"))
    _assert_equal(restored, record)
    assert restored.token_tags.device.type == "cpu"
    manifest = MiniMaxH3CacheManifest(**format.model_dump(), entries=(entry, entry))
    save_minimax_h3_manifest(manifest, tmp_path)
    assert load_minimax_h3_manifest(tmp_path) == manifest
    layout = build_record_layout(restored, format=format)
    assert layout.target_video.frames == (1 if single else 7)
    assert layout.target_audio_frames == (2 if single else 37)
    if task == "ref2va":
        assert [s.role for s in layout.segments][:4] == ["text", "ref_000_image", "ref_001_audio", "ref_001_video"]
    if single:
        assert layout.time_overrides is not None
        assert layout.time_overrides.target_time == 11.666666666666668
        if task == "fl2va":
            assert layout.time_overrides is not None
            assert layout.time_overrides.condition_times == (15.0, 10 / 3, 15.0)
    assert torch.equal(rng, torch.get_rng_state())
    assert original == (tmp_path / entry.file).read_bytes()
    with TensorReader(tmp_path / entry.file) as reader:
        assert not any("rcm" in key or "mask" in key for key in reader.keys())
        assert not any(key.startswith("ss_") for key in reader.metadata())


def test_identity_collision_and_failed_manifest_preserve_previous_files(tmp_path):
    format, record, identity = _fixture("t2va", False)
    entry = save_minimax_h3_record(tmp_path, identity, record, format=format)
    manifest = MiniMaxH3CacheManifest(**format.model_dump(), entries=(entry,))
    path = save_minimax_h3_manifest(manifest, tmp_path)
    before, manifest_before = (tmp_path / entry.file).read_bytes(), path.read_bytes()
    for changed in (
        replace(record, video=record.video + 1),
        replace(record, item=record.item.model_copy(update={"audio_present": False})),
    ):
        with pytest.raises(ValueError, match="collision"):
            save_minimax_h3_record(tmp_path, identity, changed, format=format)
    assert before == (tmp_path / entry.file).read_bytes()
    invalid_entry = entry.model_copy(update={"sha256": "0" * 64})
    invalid_manifest = MiniMaxH3CacheManifest(**format.model_dump(), entries=(invalid_entry,))
    with pytest.raises(ValueError, match="digest"):
        save_minimax_h3_manifest(invalid_manifest, tmp_path)
    assert path.read_bytes() == manifest_before
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("damage", ["dtype", "tags", "extra", "missing", "identity", "format", "frame_count"])
def test_corrupt_records_rejected_even_when_manifest_file_digest_is_updated(tmp_path, damage):
    format, record, identity = _fixture("ref2va", False)
    entry = save_minimax_h3_record(tmp_path, identity, record, format=format)
    path = tmp_path / entry.file
    with TensorReader(path) as reader:
        tensors, metadata = {key: reader.read(key).clone() for key in reader.keys()}, reader.metadata()
    if damage == "dtype":
        tensors["video"] = tensors["video"].half()
    elif damage == "tags":
        tensors["token_tags"][0] = 2
    elif damage == "extra":
        tensors["rcm_mask"] = torch.ones(1)
    elif damage == "missing":
        del tensors["audio.ref_001_audio"]
    elif damage == "identity":
        metadata["unitunner.cache.identity"] = identity.model_copy(update={"source_sha256": "4" * 64}).model_dump_json()
    elif damage == "format":
        metadata["unitunner.cache.format"] = MiniMaxH3CacheFormat(task="t2va").model_dump_json()
    else:
        metadata["unitunner.cache.item"] = record.item.model_copy(update={"frame_count": 5}).model_dump_json()
    save_tensors(tensors, path, metadata=metadata)
    with pytest.raises(ValueError):
        read_minimax_h3_record(
            tmp_path, entry.model_copy(update={"sha256": file_sha256(path)}), format=format, device=torch.device("cpu")
        )


def test_mismatched_repair_policy_and_tensor_shapes_rejected(tmp_path):
    format, record, identity = _fixture("t2va", False)
    with pytest.raises(ValidationError):
        MiniMaxH3CacheFormat.model_validate({"task": "t2va", "audio_window_accounting": "repair-start-only"})
    with pytest.raises(ValidationError):
        MiniMaxH3CacheItem(frame_count=1, control_indices=(True,))
    with pytest.raises(ValidationError):
        MiniMaxH3CacheItem.model_validate({"frame_count": 1, "rcm_mask": True})
    for changed in (
        replace(record, audio=record.audio[:, :, :-1]),
        replace(record, video=record.video[:, :, :-1]),
        replace(record, visual_conditions=(record.video[:, :1],)),
        replace(record, audio_conditions=(record.audio,)),
        replace(record, text=record.text[:, :-1]),
        replace(record, video=record.video + float("nan")),
        replace(record, item=record.item.model_copy(update={"target_index": 1})),
    ):
        with pytest.raises(ValueError):
            save_minimax_h3_record(tmp_path, identity, changed, format=format)
    with pytest.raises(ValueError, match="layout"):
        save_minimax_h3_record(tmp_path, identity.model_copy(update={"layout": "unversioned"}), record, format=format)
    assert not list(tmp_path.iterdir())


def test_reference_and_single_frame_inventory_rejections(tmp_path):
    for single in (False, True):
        format, record, identity = _fixture("ref2va", single)
        for changed in (
            replace(record, visual_conditions=record.visual_conditions[:-1]),
            replace(record, audio_conditions=()),
            replace(record, audio_conditions=(record.audio_conditions[0][:, :, :-1], *record.audio_conditions[1:])),
        ):
            with pytest.raises(ValueError):
                save_minimax_h3_record(tmp_path, identity, changed, format=format)
    format, record, identity = _fixture("fl2va", True)
    for fields in ({"target_index": None}, {"audio_present": True}, {"crop_start_frame": 2}, {"control_indices": (1,)}):
        with pytest.raises(ValueError):
            save_minimax_h3_record(
                tmp_path, identity, replace(record, item=record.item.model_copy(update=fields)), format=format
            )
