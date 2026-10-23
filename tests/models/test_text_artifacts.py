"""Check standalone topology resolution and complete cache identities offline."""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig

from unitunner.models.components.text_artifacts import text_artifact_sha256, text_config_path
from unitunner.models.components.text_loading import text_weight_path


@pytest.mark.parametrize(
    ("family", "version", "model_type"),
    [
        ("z-image", None, "qwen3"),
        ("flux", None, "t5"),
        ("flux-clip", None, "clip_text_model"),
        ("flux2", "dev", "mistral3"),
        ("flux2", "klein-4b", "qwen3"),
        ("flux2", "klein-9b", "qwen3"),
        ("flux3", None, "qwen3_vl"),
        ("krea2", None, "qwen3_vl"),
        ("qwen-image", None, "qwen2_5_vl"),
        ("qwen-image-2-1", None, "qwen3_vl"),
    ],
)
def test_bare_weight_defaults_are_typed_local_configs(tmp_path, family, version, model_type):
    path = tmp_path / "arbitrary-name.safetensors"
    path.touch()
    config = AutoConfig.from_pretrained(str(text_config_path(path, family, version=version)), local_files_only=True)
    assert config.model_type == model_type
    if family == "z-image":
        assert config.hidden_size == 2560 and config.num_hidden_layers == 36
    assert text_weight_path(path) == path


def test_sidecars_override_defaults_and_change_identity(tmp_path):
    path = tmp_path / "qwen.safetensors"
    save_file({"weight": torch.ones(1)}, path)
    original = text_artifact_sha256(path, "z-image")
    common = tmp_path / "config.json"
    common.write_text(json.dumps({"model_type": "qwen3", "hidden_size": 16}))
    assert text_config_path(path, "z-image") == common
    configured = text_artifact_sha256(path, "z-image")
    assert configured != original
    specific = path.with_suffix(".json")
    specific.write_text(common.read_text())
    assert text_config_path(path, "z-image") == specific
    assert text_artifact_sha256(path, "z-image") == configured
    specific.write_text(json.dumps({"model_type": "qwen3", "hidden_size": 32}))
    assert text_artifact_sha256(path, "z-image") != configured


def test_cache_identity_covers_every_shard_and_rejects_missing_shards(tmp_path):
    first = tmp_path / "text-00001-of-00002.safetensors"
    second = tmp_path / "text-00002-of-00002.safetensors"
    save_file({"a": torch.ones(1)}, first)
    save_file({"b": torch.ones(1)}, second)
    before = text_artifact_sha256(first, "z-image")
    assert before == text_artifact_sha256(second, "z-image")
    save_file({"b": torch.zeros(1)}, second)
    assert text_artifact_sha256(first, "z-image") != before
    second.unlink()
    with pytest.raises(FileNotFoundError):
        text_artifact_sha256(first, "z-image")


def test_variable_topology_requires_config_and_missing_paths_fail_clearly(tmp_path):
    path = tmp_path / "text.safetensors"
    with pytest.raises(FileNotFoundError, match="Text encoder does not exist"):
        text_config_path(path, "moss-image")
    path.touch()
    with pytest.raises(ValueError, match="single-file text encoder requires"):
        text_config_path(path, "moss-image")
    config = tmp_path / "config.json"
    config.write_text('{"model_type": "qwen3_5_text"}')
    assert text_config_path(path, "moss-image") == config
