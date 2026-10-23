"""Reject invalid H3 text loading choices before processor or weight IO."""

import json

import pytest
import torch

from tests.support.assertions import invalid_input
from unitunner.models.minimax_h3.text.loading import load_text_encoder


def test_invalid_text_execution_choices_fail_before_file_io(tmp_path):
    options = {"config_directory": tmp_path, "processor_directory": tmp_path, "device": torch.device("cpu")}
    with pytest.raises(ValueError, match="FP32/BF16/FP16"):
        load_text_encoder(tmp_path / "absent.safetensors", **options, dtype=torch.int8)
    with pytest.raises(ValueError, match="source_format"):
        load_text_encoder(tmp_path / "absent.safetensors", **options, source_format=invalid_input("automatic"))


def test_invalid_text_geometry_fails_before_processor_and_weights(tmp_path):
    config = {
        "model_type": "qwen3_vl",
        "text_config": {"hidden_size": 5120, "num_hidden_layers": 64},
        "vision_config": {"out_hidden_size": 5120},
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
    }
    options = {"config_directory": tmp_path, "processor_directory": tmp_path, "device": torch.device("cpu")}
    config["text_config"]["hidden_size"] = 16
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="widths"):
        load_text_encoder(tmp_path / "absent.safetensors", **options)
    config["text_config"]["hidden_size"] = 5120
    config["text_config"]["num_hidden_layers"] = 49
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="at least50"):
        load_text_encoder(tmp_path / "absent.safetensors", **options)
    config["text_config"]["num_hidden_layers"] = 64
    config["image_token_id"] = 7
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="vision token IDs"):
        load_text_encoder(tmp_path / "absent.safetensors", **options)
