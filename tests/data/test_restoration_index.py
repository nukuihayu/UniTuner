"""Validate restoration sources once without retaining payloads or changing resume hashes."""

import hashlib

import pytest
import torch

from unitunner.artifacts.hashing import canonical_sha256, file_sha256
from unitunner.data.cache import storage
from unitunner.data.cache.schema import CacheIdentity
from unitunner.data.datasets.restoration import index_restoration_pairs
from unitunner.data.degradation.config import DegradationConfig


@pytest.fixture
def dataset(tmp_path):
    high, low, entries, records = [], [], [], []
    cache = tmp_path / "cache"
    for index in range(3):
        hq, lq = tmp_path / f"{index}.png", tmp_path / f"{index}.jpg"
        hq.write_bytes(f"high {index}".encode())
        lq.write_bytes(f"low {index}".encode())
        caption = f"caption {index} 雪"
        hq.with_suffix(".txt").write_text(caption, encoding="utf-8")
        source = canonical_sha256(
            {"image": file_sha256(hq), "caption_utf8": hashlib.sha256(caption.encode()).hexdigest()}
        )
        identity = CacheIdentity(
            source_sha256=source, encoder_sha256="a" * 64, transform_sha256="b" * 64, layout="z-image/CHW+LD"
        )
        entry = storage.save_latent_record(cache, identity, torch.zeros(2, 2, 2), torch.zeros(2, 4))
        entries.append(entry)
        high.append(hq)
        low.append(lq)
        records.append({"high": source, "low": file_sha256(lq), "cache": entry.sha256})
    storage.save_latent_manifest(storage.LatentManifest(entries=tuple(entries)), cache)
    return tuple(high), tuple(low), cache, records


@pytest.mark.parametrize("synthetic", (False, True))
def test_streaming_index_preserves_legacy_digest_and_validates_each_header_once(dataset, monkeypatch, synthetic):
    high, low, cache, records = dataset
    degradation = DegradationConfig(enabled=True) if synthetic else None
    if synthetic:
        records = [{**row, "low": None} for row in records]
    reader = storage.TensorReader
    opened = []
    scanned = []

    def scan(path, *, release_cache=False):
        assert release_cache, "Restoration preflight must release scanned file pages"
        scanned.append(path)
        return file_sha256(path, release_cache=release_cache)

    def open_header(path):
        opened.append(path)
        return reader(path)

    def no_payload(*args, **kwargs):
        raise AssertionError("Indexing must never materialize cached tensors")

    monkeypatch.setattr(storage, "TensorReader", open_header)
    monkeypatch.setattr(reader, "read", no_payload)
    monkeypatch.setattr(storage, "file_sha256", scan)
    monkeypatch.setattr("unitunner.data.datasets.restoration.file_sha256", scan)
    pairs, digest = index_restoration_pairs(
        high, None if synthetic else low, cache, family="z-image", degradation=degradation
    )
    assert len(opened) == len(high)
    assert len(scanned) == len(high) * (2 if synthetic else 3)
    assert [pair.high for pair in pairs] == list(high)
    assert [pair.low for pair in pairs] == ([None] * len(high) if synthetic else list(low))
    assert digest == canonical_sha256(
        {
            "schema": "restoration-images-random-delivery-3" if synthetic else "restoration-images-2",
            "entries": records,
            "degradation": None if degradation is None else degradation.model_dump(mode="json"),
        }
    )


@pytest.mark.parametrize("changed", ("image", "caption", "record", "count", "duplicate", "family", "layout"))
def test_index_still_rejects_invalid_sources_and_caches(dataset, changed):
    high, low, cache, _ = dataset
    family = "z-image"
    if changed == "image":
        high[0].write_bytes(b"changed")
    elif changed == "caption":
        high[0].with_suffix(".txt").write_text("changed")
    elif changed == "record":
        manifest = storage.read_latent_manifest(cache)
        with (cache / manifest.entries[0].file).open("ab") as output:
            output.write(b"corrupted")
    elif changed == "count":
        high, low = high[:-1], low[:-1]
    elif changed == "duplicate":
        high = (high[0], high[0], high[2])
    elif changed == "family":
        family = "flux"
    else:
        manifest = storage.read_latent_manifest(cache)
        entry = manifest.entries[0]
        with storage.TensorReader(cache / entry.file) as reader:
            identity = CacheIdentity.model_validate_json(reader.metadata()["unitunner.cache.identity"])
        wrong = storage.save_cache_record(
            cache, identity.model_copy(update={"transform_sha256": "c" * 64}), {"unknown": torch.zeros(1)}
        )
        storage.save_latent_manifest(manifest.model_copy(update={"entries": (wrong, *manifest.entries[1:])}), cache)
    with pytest.raises(ValueError):
        index_restoration_pairs(high, low, cache, family=family)
