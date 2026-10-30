"""Verify native encoders acquire only tokenizer assets from pinned Hub sources."""

from fnmatch import fnmatch
from pathlib import Path

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic import ValidationError

from unitunner.config.text import TextEncodingConfig, TokenizerSourceConfig
from unitunner.models import text_assets


@pytest.mark.parametrize(
    "family,version,role,expected",
    [
        ("z-image", None, "tokenizer", ("Tongyi-MAI/Z-Image", "tokenizer")),
        ("flux", None, "tokenizer", ("google/t5-v1_1-xxl", None)),
        ("flux", None, "clip_tokenizer", ("openai/clip-vit-large-patch14", None)),
        ("flux2", "dev", "tokenizer", ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", None)),
        ("flux2", "klein-4b", "tokenizer", ("Qwen/Qwen3-4B", None)),
        ("flux2", "klein-9b", "tokenizer", ("Qwen/Qwen3-8B", None)),
        ("qwen-image", None, "tokenizer", ("Qwen/Qwen-Image", "tokenizer")),
        ("qwen-image", None, "processor", ("Qwen/Qwen-Image-Edit", "processor")),
        ("krea2", None, "tokenizer", ("Qwen/Qwen3-VL-4B-Instruct", None)),
        ("wan", None, "tokenizer", ("Wan-AI/Wan2.1-T2V-14B", "google/umt5-xxl")),
        ("minimax-h3", None, "tokenizer", ("MiniMaxAI/MiniMax-H3", "processor")),
    ],
)
def test_musubi_defaults(family, version, role, expected):
    assert text_assets.default_tokenizer(family, version, role=role) == expected


def test_hub_filters_revision_and_offline(monkeypatch, tmp_path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(text_assets, "snapshot_download", download)
    config = TextEncodingConfig(tokenizer=TokenizerSourceConfig(revision="immutable-sha", offline=True))
    assert text_assets.tokenizer_directory(config, "wan") == tmp_path / "google/umt5-xxl"
    call = calls.pop()
    assert call["revision"] == "immutable-sha"
    assert call["local_files_only"] is True
    patterns = call["allow_patterns"]
    assert any(fnmatch("google/umt5-xxl/spiece.model", pattern) for pattern in patterns)
    assert any(fnmatch("google/umt5-xxl/config.json", pattern) for pattern in patterns)
    for filename in ("model.safetensors", "pytorch_model.bin", "model-00001-of-00020.safetensors", "modeling.py"):
        assert not any(fnmatch("google/umt5-xxl/" + filename, pattern) for pattern in patterns)


def test_local_override_stays_offline(monkeypatch, tmp_path):
    def unexpected(**kwargs):
        pytest.fail("Local tokenizer attempted Hub access")

    monkeypatch.setattr(text_assets, "snapshot_download", unexpected)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "tokenizer.json").write_text("{}")
    config = TextEncodingConfig(tokenizer=TokenizerSourceConfig(source=str(tmp_path), subfolder="assets"))
    pinned, _ = text_assets.pin_tokenizer_sources(config, "z-image")
    assert pinned.tokenizer.source is not None
    assert Path(pinned.tokenizer.source) == assets
    assert pinned.tokenizer.offline is True


def test_qwen_and_flux_sources_are_independent(monkeypatch, tmp_path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs["repo_id"])
        return str(tmp_path)

    monkeypatch.setattr(text_assets, "snapshot_download", download)
    config = TextEncodingConfig()
    text_assets.tokenizer_directory(config, "qwen-image")
    text_assets.tokenizer_directory(config, "qwen-image", role="processor")
    text_assets.tokenizer_directory(config, "flux")
    text_assets.tokenizer_directory(config, "flux", role="clip_tokenizer")
    assert calls == ["Qwen/Qwen-Image", "Qwen/Qwen-Image-Edit", "google/t5-v1_1-xxl", "openai/clip-vit-large-patch14"]


def test_offline_failure_is_actionable(monkeypatch):
    def missing(**kwargs):
        raise LocalEntryNotFoundError("Snapshot not cached")

    monkeypatch.setattr(text_assets, "snapshot_download", missing)
    config = TextEncodingConfig(tokenizer=TokenizerSourceConfig(offline=True))
    with pytest.raises(ValueError, match=r"Tongyi-MAI/Z-Image.*offline cache"):
        text_assets.tokenizer_directory(config, "z-image")


@pytest.mark.parametrize("subfolder", ["../weights", "/absolute", "a/../../b", "a\\b", "C:/weights"])
def test_subfolder_cannot_escape_source(subfolder):
    with pytest.raises(ValidationError, match="relative repository path"):
        TokenizerSourceConfig(subfolder=subfolder)


def test_missing_local_source_does_not_become_hub_id(tmp_path):
    config = TextEncodingConfig(tokenizer=TokenizerSourceConfig(source=str(tmp_path / "missing")))
    with pytest.raises(ValueError, match="local directory does not exist"):
        text_assets.tokenizer_directory(config, "z-image")


@pytest.mark.parametrize("single_file", [False, True])
def test_native_weights_load_with_separate_hub_tokenizer(monkeypatch, tmp_path, single_file):
    import torch
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model

    from unitunner.models.z_image.text.encoder import load_z_image_text

    weights = tmp_path / "native"
    assets = tmp_path / "snapshot" / "tokenizer"
    vocabulary = {value: index for index, value in enumerate(sorted(ByteLevel.alphabet()))}
    vocabulary.update({"<|endoftext|>": 256, "<|im_start|>": 257, "<|im_end|>": 258})
    tokenizer = Qwen2Tokenizer(vocab=vocabulary, merges=[], pad_token="<|endoftext|>")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(assets)
    Qwen3Model(
        Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=259,
        )
    ).save_pretrained(weights)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(assets.parent)

    monkeypatch.setattr(text_assets, "snapshot_download", download)
    encoder = load_z_image_text(
        weights / "model.safetensors" if single_file else weights, device=torch.device("cpu"), dtype=torch.float32
    )
    expected = encoder.encode(["test caption"]).features
    assert expected.shape[-1] == 16
    from unitunner.artifacts.tensors import save_tensors

    # Deliberately retain the base-model config while loading a causal namespace.
    save_tensors({"model." + k: v for k, v in encoder.model.state_dict().items()}, weights / "model.safetensors")
    from tests.support.text_assets import local_text_config

    restored = load_z_image_text(
        weights, text_config=local_text_config(assets), device=torch.device("cpu"), dtype=torch.float32
    )
    torch.testing.assert_close(restored.encode(["test caption"]).features, expected, atol=0, rtol=0)
    assert calls[0]["repo_id"] == "Tongyi-MAI/Z-Image"
    assert not (weights / "tokenizer.json").exists()
    assert not (assets / "model.safetensors").exists()
