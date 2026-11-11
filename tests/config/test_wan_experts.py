"""Validate the paired Wan recipe across cache and training entry points."""

import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from unituner.config.cache import CacheConfig, cache_document
from unituner.config.training import TrainingConfig

EXAMPLE = Path(__file__).parents[2] / "docs/examples/wan-dual-expert-flow.toml"


def test_paired_example_shares_a_cache_without_loading_second_denoiser():
    document = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))
    training = TrainingConfig.model_validate_json(json.dumps(document))
    cache = CacheConfig.model_validate_json(json.dumps(cache_document(document)))
    assert training.wan_experts.boundary == 0.875
    assert cache.model.version == "2.2" and cache.model.family == "wan"


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "version", "2.1"),
        ("model", "family", "flux"),
        ("wan_experts", "boundary", 0.0),
        ("wan_experts", "boundary", 1.0),
        ("wan_experts", "high_noise_weights", "/models/wan2.2/low_noise_model.safetensors"),
        ("wan_experts", "routing", "per-sample"),
    ],
)
def test_invalid_expert_selection_fails_before_weight_io(section, key, value):
    document = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))
    document[section][key] = value
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate_json(json.dumps(document))
