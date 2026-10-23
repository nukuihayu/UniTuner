"""Verify deterministic paging, source options and preview failure isolation."""

import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from unitunner.data.cache.storage import CacheEntry
from unitunner.data.datasets.restoration import AlignedImagePair, RestorationDataset
from unitunner.training.previews.images import PreviewImages
from unitunner.training.previews.restoration import RestorationPreviews


def fixture(tmp_path, *, at_start=True, count=2):
    groups = {
        "a": SimpleNamespace(cache_directory=tmp_path / "a", resolution=(8, 8), horizontal_flip_probability=0.0),
        "b": SimpleNamespace(cache_directory=tmp_path / "b", resolution=(16, 16), horizontal_flip_probability=0.5),
    }
    entry = CacheEntry(file="a" * 64 + ".safetensors", sha256="b" * 64)
    pairs = tuple(
        AlignedImagePair(None, Path(str(i)), entry, groups["a" if i % 2 == 0 else "b"].cache_directory)
        for i in range(5)
    )
    pairs = (*pairs[:2], pairs[0], *pairs[2:])
    reads = []

    def read(pair, resolution, seed, flip):
        reads.append((pair, resolution, seed, flip))
        return torch.zeros(3, 8, 8), torch.ones(3, 8, 8)

    config = SimpleNamespace(
        train=SimpleNamespace(seed=17),
        dataset=SimpleNamespace(groups=groups),
        preview=SimpleNamespace(at_start=at_start, every_updates=10, num_images=count),
        output=SimpleNamespace(directory=tmp_path),
        execution=SimpleNamespace(precision="fp32"),
    )
    return config, RestorationDataset(pairs, "digest", read), reads


@pytest.mark.parametrize("at_start", [False, True])
@pytest.mark.parametrize("count", [1, 2, 10])
def test_pages_are_repeat_independent_and_resume_from_update(tmp_path, at_start, count):
    config, dataset, reads = fixture(tmp_path, at_start=at_start, count=count)
    before = random.getstate()
    preview = RestorationPreviews(config, dataset)
    assert random.getstate() == before
    assert not reads
    assert len(preview.samples) == 5
    expected_seeds = {pair: 17 + dataset.pairs.index(pair) for pair in dataset.pairs}
    assert {sample.pair: sample.seed for sample in preview.samples} == expected_seeds
    first_update = 0 if at_start else 10
    first = preview.select(first_update)
    second = preview.select(first_update + 10)
    combined = [sample for _, sample in first + second]
    assert combined == [preview.samples[i % 5] for i in range(2 * min(count, 5))]
    # A final update before the next cadence gets that next page.
    assert preview.select(first_update + 11) == preview.select(first_update + 20)
    restarted = RestorationPreviews(config, dataset)
    assert restarted.select(37) == preview.select(37)
    assert restarted.select(10) == preview.select(10)
    assert random.getstate() == before


def test_render_uses_original_group_options_and_publishes_comparison(tmp_path, monkeypatch):
    from unitunner.training.previews import restoration

    config, dataset, reads = fixture(tmp_path, count=10)
    preview = RestorationPreviews(config, dataset)
    written = []
    monkeypatch.setattr(restoration, "save_image", lambda pixels, path: written.append((pixels, path)))
    method = SimpleNamespace(
        captions=lambda pairs: pairs,
        encode=lambda low: low,
        restore=lambda latent, caption: latent,
        decode=lambda latent: latent,
    )
    preview(11, method, torch.device("cpu"))
    assert len(reads) == len(written) == 5
    for index, ((pair, resolution, seed, flip), (pixels, path)) in enumerate(zip(reads, written, strict=True)):
        group = config.dataset.groups["a" if int(pair.high.name) % 2 == 0 else "b"]
        assert (resolution, flip) == (group.resolution, group.horizontal_flip_probability)
        assert seed == 17 + dataset.pairs.index(pair)
        assert pixels.shape == (3, 12, 32)
        assert torch.equal(pixels[:, 2:10, 22:30], torch.ones(3, 8, 8))
        assert path == tmp_path / "previews" / f"step-00000011-{index:02d}.png"


def test_preview_failure_restores_rng_and_model_modes_and_can_retry(tmp_path, monkeypatch):
    from unitunner.training.previews import restoration

    config, dataset, reads = fixture(tmp_path)
    renderer = PreviewImages(config, dataset)
    model, branches, vae = nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1)
    branches.eval()

    def fail(low):
        random.random()
        torch.rand(1)
        raise RuntimeError("encode failed")

    method = SimpleNamespace(
        vae=vae,
        encoder=None,
        captions=lambda pairs: pairs,
        encode=fail,
        restore=lambda latent, caption: latent,
        decode=lambda latent: latent,
    )
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="encode failed"):
        renderer(10, model, branches, method)
    assert (model.training, branches.training, vae.training) == (True, False, True)
    assert random.getstate() == python_rng
    assert torch.equal(torch_rng, torch.get_rng_state())
    method.encode = lambda low: low
    monkeypatch.setattr(restoration, "save_image", lambda pixels, path: None)
    renderer(10, model, branches, method)
    assert reads[0] == reads[1]


def test_invalid_preview_sources_fail_without_reading_images(tmp_path):
    config, dataset, reads = fixture(tmp_path)
    with pytest.raises(ValueError, match="nonempty"):
        RestorationPreviews(config, RestorationDataset((), "digest", dataset.read))
    config.dataset.groups = {}
    with pytest.raises(ValueError, match="No restoration dataset group"):
        RestorationPreviews(config, dataset)
    assert not reads
