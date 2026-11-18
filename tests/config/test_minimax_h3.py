"""Validate H3 native AV geometry and encoder choices before artifact access."""

from pathlib import Path

import pytest
from pydantic import JsonValue

from unituner.checkpoints.schema import build_resume_identity
from unituner.config.loading import ConfigError, load_cache_config, load_training_config, merge_config_overrides

EXAMPLE = Path(__file__).resolve().parents[2] / "docs/examples/minimax-h3-cache.toml"


def test_h3_explicit_video_and_single_frame_config():
    config = load_cache_config(EXAMPLE)
    assert config.model.minimax_h3 is not None
    assert config.model.minimax_h3.text_device == "cpu"
    assert config.dataset.groups["main"].video is not None
    assert config.dataset.groups["main"].video.target_frames == (124,)
    single = load_cache_config(EXAMPLE, overrides={"dataset": {"groups": {"main": {"kind": "image", "video": None}}}})
    assert single.dataset.groups["main"].kind == "image" and single.dataset.groups["main"].video is None
    with pytest.raises(ConfigError, match="MiniMaxH3"):
        load_cache_config(EXAMPLE, overrides={"model": {"family": "z-image", "variant": "base"}})
    with pytest.raises(ConfigError, match="24Hz"):
        load_cache_config(EXAMPLE, overrides={"dataset": {"groups": {"main": {"video": {"target_fps": 16.0}}}}})
    with pytest.raises(ConfigError, match="17"):
        load_cache_config(EXAMPLE, overrides={"dataset": {"groups": {"main": {"video": {"target_frames": [81]}}}}})
    with pytest.raises(ConfigError, match="32kHz"):
        load_cache_config(
            EXAMPLE,
            overrides={
                "dataset": {"groups": {"main": {"minimax_h3": {"audio_decode": {"sample_rate": 44100, "channels": 2}}}}}
            },
        )
    with pytest.raises(ConfigError, match="Extra inputs"):
        load_cache_config(EXAMPLE, overrides={"dataset": {"groups": {"main": {"minimax_h3": {"rcm_mask": True}}}}})


def test_h3_flow_band_validation_and_method_ownership():
    training = EXAMPLE.with_name("minimax-h3-train.toml")
    config = load_training_config(training)
    assert config.method.minimax_h3 is not None
    assert config.method.minimax_h3.video_shift == 12.0
    with pytest.raises(ConfigError, match="inside"):
        load_training_config(
            training,
            overrides={"method": {"minimax_h3": {"sigma_min": 0.2, "focus_min": 0.1, "focus_probability": 0.5}}},
        )
    with pytest.raises(ConfigError, match="100"):
        load_training_config(training, overrides={"method": {"minimax_h3": {"video_shift": 101.0}}})
    with pytest.raises(ConfigError, match="requires MiniMaxH3"):
        load_training_config(training, overrides={"model": {"family": "z-image", "variant": "base"}})
    with pytest.raises(ConfigError, match="Extra inputs"):
        load_training_config(training, overrides={"method": {"minimax_h3": {"rcm_mask": True}}})


def test_h3_preview_contract_and_resume_identity():
    def training_config(overrides):
        base = {"model": {"weights": "/models/h3/transformer.safetensors"}, "train": {"adapter": {}}}
        return load_training_config(EXAMPLE, overrides=merge_config_overrides(base, overrides))

    config = training_config({"model": {"variant": "t2va"}})
    preview: dict[str, JsonValue] = {"audio_video": [{"inputs": {"prompt": "Hello"}, "sampling": {"frames": 1}}]}
    enabled = training_config({"model": {"variant": "t2va"}, "preview": preview})

    def identity(value):
        return build_resume_identity(value, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64)

    assert identity(config) == identity(enabled)
    for invalid in ({}, {**preview, "samples": [{"prompt": "hello"}]}, {**preview, "num_images": 2}):
        with pytest.raises(ConfigError, match=r"preview\.audio_video"):
            training_config({"preview": invalid})
    with pytest.raises(ConfigError, match="require MiniMaxH3"):
        load_training_config(EXAMPLE.with_name("flow.toml"), overrides={"preview": preview})
    with pytest.raises(ConfigError, match="FL2VA"):
        training_config({"model": {"variant": "fl2va"}, "preview": preview})
