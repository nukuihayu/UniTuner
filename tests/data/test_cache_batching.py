"""Exercise native batched image/text encoding, masks and immutable cache publication."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Qwen3Config, Qwen3Model

from unituner.data.cache.images import cache_images
from unituner.data.cache.storage import load_latent_manifest, read_latent_record, save_latent_manifest
from unituner.data.media.buckets import ImageBuckets
from unituner.models.z_image.config.vae import ZImageVAEConfig
from unituner.models.z_image.text.encoder import ZImageTextEncoder
from unituner.models.z_image.vae import ZImageVAE


class TinyTokenizer:
    padding_side = "right"

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, prompts, *, max_length, **kwargs):
        ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, prompt in enumerate(prompts):
            length = min(len(prompt) + 1, max_length)
            ids[index, :length] = torch.arange(1, length + 1)
            mask[index, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


@pytest.fixture
def encoders():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(12)
        vae = (
            ZImageVAE(ZImageVAEConfig(block_channels=(8, 16), norm_groups=4, layers_per_block=1, latent_channels=2))
            .eval()
            .requires_grad_(False)
        )
        model = (
            Qwen3Model(
                Qwen3Config(
                    vocab_size=64,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    head_dim=16,
                )
            )
            .eval()
            .requires_grad_(False)
        )
    try:
        yield vae, ZImageTextEncoder(TinyTokenizer(), model, max_length=16)
    finally:
        torch.set_num_threads(previous)


def source_images(tmp_path, sizes):
    source = tmp_path / "images"
    source.mkdir()
    for index, (width, height) in enumerate(sizes):
        pixels = np.random.default_rng(index).integers(0, 256, (height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(source / f"{index}.png")
        (source / f"{index}.txt").write_text("x" * index)
    return source


@pytest.mark.parametrize("batch_size", (2, 16))
def test_batched_native_outputs_masks_tail_reuse_and_rng(tmp_path, encoders, monkeypatch, batch_size):
    vae, text = encoders
    source = source_images(tmp_path, [(32, 32)] * 5)
    single, batched = tmp_path / "single", tmp_path / "batched"
    options = {"resolution": (32, 32), "encoder_sha256": "a" * 64}
    rng = torch.get_rng_state().clone()
    baseline = cache_images(source, single, vae, text, **options)
    vae_calls, text_calls = [], []
    encode_vae, encode_text = vae.encode, text.encode

    def vae_batch(pixels):
        vae_calls.append(pixels.shape[0])
        return encode_vae(pixels)

    def text_batch(prompts):
        text_calls.append(len(prompts))
        return encode_text(prompts)

    monkeypatch.setattr(vae, "encode", vae_batch)
    monkeypatch.setattr(text, "encode", text_batch)
    manifest = cache_images(source, batched, vae, text, batch_size=batch_size, **options)
    assert vae_calls == text_calls == ([2, 2, 1] if batch_size == 2 else [5])
    assert torch.equal(torch.get_rng_state(), rng)
    for index, (expected, actual) in enumerate(zip(baseline.entries, manifest.entries, strict=True)):
        left = read_latent_record(single, expected, device=torch.device("cpu"))
        right = read_latent_record(batched, actual, device=torch.device("cpu"))
        assert right.caption.shape == (index + 1, 32)
        torch.testing.assert_close(right.latents, left.latents, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(right.caption, left.caption, atol=1e-6, rtol=1e-5)
        assert actual.file != expected.file
    vae_calls.clear()
    text_calls.clear()
    assert cache_images(source, batched, vae, text, batch_size=batch_size, **options) == manifest
    assert vae_calls == text_calls == []


def test_bucket_batches_keep_order_and_publish_repa_per_image(tmp_path, encoders, monkeypatch):
    vae, text = encoders
    source = source_images(tmp_path, [(32, 16), (32, 16), (16, 32), (16, 32), (32, 16)])
    buckets = ImageBuckets((32, 32), no_upscale=True)
    shapes, repa_records = [], []
    encode = vae.encode

    def capture(pixels):
        shapes.append(tuple(pixels.shape))
        return encode(pixels)

    monkeypatch.setattr(vae, "encode", capture)
    manifest = cache_images(
        source,
        tmp_path / "cache",
        vae,
        text,
        resolution=(32, 32),
        encoder_sha256="a" * 64,
        buckets=buckets,
        batch_size=16,
        repa=SimpleNamespace(publish_normalized=lambda entry, pixels: repa_records.append((entry, pixels.shape))),
    )
    assert shapes == [(2, 3, 16, 32), (2, 3, 32, 16), (1, 3, 16, 32)]
    assert [entry for entry, _ in repa_records] == list(manifest.entries)
    assert [shape for _, shape in repa_records] == [
        (3, 1, 16, 32),
        (3, 1, 16, 32),
        (3, 1, 32, 16),
        (3, 1, 32, 16),
        (3, 1, 16, 32),
    ]


def test_batch_failure_keeps_previous_manifest_and_retry_recovers(tmp_path, encoders, monkeypatch):
    vae, text = encoders
    source = source_images(tmp_path, [(32, 32)] * 3)
    destination = tmp_path / "cache"
    options = {"resolution": (32, 32), "encoder_sha256": "a" * 64}
    baseline = cache_images(source, destination, vae, text, **options)
    encode = text.encode
    calls = 0

    def fail_second(prompts):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("text failed")
        return encode(prompts)

    monkeypatch.setattr(text, "encode", fail_second)
    with pytest.raises(RuntimeError, match="text failed"):
        cache_images(source, destination, vae, text, batch_size=2, **options)
    assert load_latent_manifest(destination) == baseline
    monkeypatch.setattr(text, "encode", encode)
    restored = cache_images(source, destination, vae, text, batch_size=2, **options)
    assert len(restored.entries) == 3
    assert load_latent_manifest(destination) == restored


def test_partial_reuse_retains_peers_and_changed_prompt_invalidates_its_batch(tmp_path, encoders, monkeypatch):
    vae, text = encoders
    source = source_images(tmp_path, [(32, 32)] * 3)
    destination = tmp_path / "cache"
    options = {"resolution": (32, 32), "encoder_sha256": "a" * 64, "batch_size": 2}
    original = cache_images(source, destination, vae, text, **options)
    save_latent_manifest(original.model_copy(update={"entries": original.entries[1:]}), destination)
    calls = []
    encode = text.encode

    def capture(prompts):
        calls.append(prompts)
        return encode(prompts)

    monkeypatch.setattr(text, "encode", capture)
    assert cache_images(source, destination, vae, text, **options) == original
    assert calls == [["", "x"]]
    calls.clear()
    (source / "0.txt").write_text("changed")
    changed = cache_images(source, destination, vae, text, **options)
    assert calls == [["changed", "x"]]
    assert all(a.file != b.file for a, b in zip(original.entries[:2], changed.entries[:2], strict=True))
    assert original.entries[-1] == changed.entries[-1]
