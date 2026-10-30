"""Validate restoration sources without reading cached tensor payloads."""

import hashlib

import pytest
import torch

from unituner.artifacts.hashing import canonical_sha256, file_sha256
from unituner.data.cache import storage
from unituner.data.cache.schema import CacheIdentity
from unituner.data.datasets.restoration import index_restoration_pairs
from unituner.data.degradation.config import DegradationConfig


@pytest.fixture
def dataset(tmp_path):
    high, low, entries = [], [], []
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
    storage.save_latent_manifest(storage.LatentManifest(entries=tuple(entries)), cache)
    return tuple(high), tuple(low), cache


@pytest.mark.parametrize("synthetic", (False, True))
@pytest.mark.parametrize("workers", (1, 4))
def test_streaming_index_validates_each_header_once(dataset, monkeypatch, synthetic, workers):
    high, low, cache = dataset
    degradation = DegradationConfig(enabled=True) if synthetic else None
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
    monkeypatch.setattr("unituner.data.datasets.restoration.file_sha256", scan)
    pairs, _ = index_restoration_pairs(
        high, None if synthetic else low, cache, family="z-image", degradation=degradation, validation_workers=workers
    )
    assert len(opened) == len(high)
    assert len(scanned) == len(high) * (2 if synthetic else 3)
    assert [pair.high for pair in pairs] == list(high)
    assert [pair.low for pair in pairs] == ([None] * len(high) if synthetic else list(low))


@pytest.mark.parametrize("changed", ("image", "caption", "record", "count", "duplicate", "family", "layout"))
def test_index_still_rejects_invalid_sources_and_caches(dataset, changed):
    high, low, cache = dataset
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
            identity = CacheIdentity.model_validate_json(reader.metadata()["unituner.cache.identity"])
        wrong = storage.save_cache_record(
            cache, identity.model_copy(update={"transform_sha256": "c" * 64}), {"unknown": torch.zeros(1)}
        )
        storage.save_latent_manifest(manifest.model_copy(update={"entries": (wrong, *manifest.entries[1:])}), cache)
    with pytest.raises(ValueError):
        index_restoration_pairs(high, low, cache, family=family)


@pytest.mark.parametrize("synthetic", (False, True))
def test_trusted_index_skips_full_scans(dataset, monkeypatch, synthetic):
    high, low, cache = dataset
    degradation = DegradationConfig(enabled=True) if synthetic else None
    kwargs = {"family": "z-image", "degradation": degradation}
    expected, _ = index_restoration_pairs(high, None if synthetic else low, cache, **kwargs)
    scans = []

    def scan(path, **options):
        assert path in low, "Trusted indexing must not hash HQ images or cache payloads"
        scans.append(path)
        return file_sha256(path, **options)

    def no_validation(*args, **kwargs):
        raise AssertionError("Full record validation must be skipped")

    # Trusted mode deliberately accepts changed HQ/captions, using cached source
    # identities. The caller promises those contents still match their caches.
    high[0].write_bytes(b"changed")
    high[0].with_suffix(".txt").write_text("changed")
    monkeypatch.setattr("unituner.data.datasets.restoration.file_sha256", scan)
    monkeypatch.setattr("unituner.data.datasets.restoration.validate_latent_entry", no_validation)
    monkeypatch.setattr(storage.TensorReader, "read", no_validation)
    actual, _ = index_restoration_pairs(high, None if synthetic else low, cache, skip_validation=True, **kwargs)
    assert actual == expected
    assert sorted(scans) == ([] if synthetic else sorted(low))


@pytest.mark.parametrize("changed", ("count", "duplicate", "family"))
def test_trusted_index_keeps_structural_checks(dataset, changed):
    high, low, cache = dataset
    family = "z-image"
    if changed == "count":
        high, low = high[:-1], low[:-1]
    elif changed == "duplicate":
        high = (high[0], high[0], high[2])
    else:
        family = "flux"
    with pytest.raises(ValueError):
        index_restoration_pairs(high, low, cache, family=family, skip_validation=True)


def test_trusted_cached_image_index_skips_hashes_but_preserves_batches(dataset, monkeypatch):
    from unituner.config.datasets import DataConfig, DatasetConfig
    from unituner.data.datasets import cached_images

    _, _, cache = dataset
    config = DataConfig(groups={"main": DatasetConfig(source="unused", cache_directory=str(cache))})
    kwargs = {"family": "z-image", "variant": "turbo", "version": None}
    expected = cached_images.index_cached_images(config, **kwargs)

    def no_validation(*args, **kwargs):
        raise AssertionError("Full cache validation must be skipped")

    monkeypatch.setattr(cached_images, "load_latent_manifest", no_validation)
    assert cached_images.index_cached_images(config, skip_validation=True, **kwargs) == expected
    with pytest.raises(ValueError, match="family/version"):
        cached_images.index_cached_images(config, family="flux", variant="dev", version=None, skip_validation=True)
