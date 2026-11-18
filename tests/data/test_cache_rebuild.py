"""Verify fresh cache publication, preservation and multi-directory rollback."""

from pathlib import Path

import pytest

from unituner.data.cache.rebuild import rebuild_directories


def write_cache(path, content):
    path.mkdir(exist_ok=True)
    (path / "manifest.json").write_text(content)
    (path / ("a" * 64 + ".safetensors")).write_text(content)


def test_rebuild_replaces_managed_files_preserves_unrelated_files(tmp_path):
    target = tmp_path / "cache"
    write_cache(target, "old")
    (target / "notes.txt").write_text("keep")
    (target / "nested").mkdir()
    (target / "nested/notes.txt").write_text("keep nested")
    with rebuild_directories([target]) as outputs:
        fresh = outputs[target]
        assert list(fresh.iterdir()) == []
        assert (target / "manifest.json").read_text() == "old"
        (fresh / "manifest.json").write_text("new")
        (fresh / ("b" * 64 + ".safetensors")).write_text("new")
    assert (target / "manifest.json").read_text() == "new"
    assert not (target / ("a" * 64 + ".safetensors")).exists()
    assert (target / "notes.txt").read_text() == "keep"
    assert (target / "nested/notes.txt").read_text() == "keep nested"
    assert list(tmp_path.glob(".cache-rebuild-*")) == []


def test_encoding_failure_keeps_old_records_and_manifest(tmp_path):
    target = tmp_path / "cache"
    write_cache(target, "old")
    with pytest.raises(ValueError, match="encoder failed"), rebuild_directories([target]) as outputs:
        write_cache(outputs[target], "partial")
        raise ValueError("encoder failed")
    assert (target / "manifest.json").read_text() == "old"
    assert (target / ("a" * 64 + ".safetensors")).read_text() == "old"
    assert list(tmp_path.glob(".cache-rebuild-*")) == []


def test_publish_failure_rolls_back_all_directories(tmp_path, monkeypatch):
    targets = [tmp_path / "one", tmp_path / "two"]
    for path in targets:
        write_cache(path, "old")
    rename = Path.rename

    def fail_second_publish(self, target):
        if self.name == "fresh" and target == targets[1]:
            raise OSError("publish failed")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_second_publish)
    with pytest.raises(OSError, match="publish failed"), rebuild_directories(targets) as outputs:
        for path in targets:
            write_cache(outputs[path], "new")
    for path in targets:
        assert (path / "manifest.json").read_text() == "old"
        assert (path / ("a" * 64 + ".safetensors")).read_text() == "old"
    assert list(tmp_path.glob(".*-rebuild-*")) == []


def test_overlapping_outputs_and_missing_manifest_fail_before_replacement(tmp_path):
    target = tmp_path / "cache"
    write_cache(target, "old")
    with pytest.raises(ValueError, match="distinct"), rebuild_directories([target, target / "nested"]):
        pass
    with pytest.raises(ValueError, match="no manifest"), rebuild_directories([target]):
        pass
    assert (target / "manifest.json").read_text() == "old"
