"""Keep integrity checks independent of optional kernel page-cache advice."""

import hashlib
import os

import pytest

from unitunner.artifacts.hashing import file_sha256


@pytest.mark.parametrize("release_cache", (False, True))
def test_hash_advises_only_when_requested_after_reading(tmp_path, monkeypatch, release_cache):
    path = tmp_path / "payload"
    content = b"image and caption cache" * 20000
    path.write_bytes(content)
    calls = []

    def advise(fd, offset, length, advice):
        assert os.lseek(fd, 0, os.SEEK_CUR) == len(content)
        calls.append((offset, length, advice))

    monkeypatch.setattr(os, "posix_fadvise", advise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    assert file_sha256(path, release_cache=release_cache) == hashlib.sha256(content).hexdigest()
    assert calls == ([(0, 0, 4)] if release_cache else [])


@pytest.mark.parametrize("unsupported", ("function", "constant", "filesystem"))
def test_hash_survives_unavailable_cache_advice(tmp_path, monkeypatch, unsupported):
    path = tmp_path / "payload"
    path.write_bytes(b"verified")

    def advise(*args):
        raise OSError("Unsupported filesystem")

    monkeypatch.setattr(os, "posix_fadvise", advise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    if unsupported == "function":
        monkeypatch.delattr(os, "posix_fadvise")
    elif unsupported == "constant":
        monkeypatch.delattr(os, "POSIX_FADV_DONTNEED")
    assert file_sha256(path, release_cache=True) == hashlib.sha256(b"verified").hexdigest()
