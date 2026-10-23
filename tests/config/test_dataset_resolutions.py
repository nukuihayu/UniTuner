"""Verify multiscale expansion is identical for cache and training boundaries."""

import json
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from unitunner.config.datasets import CacheDataConfig, DataConfig
from unitunner.config.loading import ConfigError, load_cache_config, load_training_config

EXAMPLES = Path(__file__).resolve().parents[2] / "docs/examples"


@pytest.mark.parametrize("schema", [CacheDataConfig, DataConfig])
@pytest.mark.parametrize("inherited", [False, True])
def test_multiscale_defaults_overrides_and_roundtrip(schema, inherited):
    resolutions = [[1024, 1024], [768, 768], [512, 512]]
    document = {
        "defaults": {"resolution": resolutions if inherited else [256, 256], "prompt": "shared caption"},
        "groups": {
            "main": {"source": "images", "cache_directory": "cache/main", "repeats": 2},
            "other": {"source": "other", "cache_directory": "cache/other", "resolution": [640, 640]},
        },
    }
    if not inherited:
        document["groups"]["main"]["resolution"] = resolutions
    config = schema.model_validate_json(json.dumps(document))
    assert list(config.groups) == ["main__1024x1024", "main__768x768", "main__512x512", "other"]
    for resolution in resolutions:
        scale = f"{resolution[0]}x{resolution[1]}"
        group = config.groups[f"main__{scale}"]
        assert group.resolution == tuple(resolution)
        assert Path(group.cache_directory) == Path("cache/main") / scale
        assert group.source == "images" and group.prompt == "shared caption" and group.repeats == 2
    assert config.groups["other"].cache_directory == "cache/other"
    assert schema.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    "resolution",
    [[], [[0, 512]], [[512, -1]], [[512, 512], [512, 512]], [[512]], [1024, [512, 512]], [[512.0, 512]], [[True, 512]]],
)
def test_invalid_multiscale_resolution_rejected(resolution):
    with pytest.raises(ValidationError):
        CacheDataConfig.model_validate_json(
            json.dumps({"groups": {"main": {"source": "images", "cache_directory": "cache", "resolution": resolution}}})
        )


def test_expanded_name_and_cache_collisions_rejected():
    main = {"source": "images", "cache_directory": "cache", "resolution": [[512, 512]]}
    for name, other in (
        ("main__512x512", {"source": "other", "cache_directory": "other"}),
        ("other", {"source": "other", "cache_directory": "cache/512x512"}),
    ):
        with pytest.raises(ValidationError, match=r"collide|distinct"):
            CacheDataConfig.model_validate_json(json.dumps({"groups": {"main": main, name: other}}))


def test_command_loaders_resolve_same_scales_without_changing_input():
    overrides: dict[str, JsonValue] = {"dataset": {"groups": {"main": {"resolution": [[1024, 1024], [512, 512]]}}}}
    before = json.dumps(overrides)
    training = load_training_config(EXAMPLES / "flow.toml", overrides=overrides)
    cache = load_cache_config(EXAMPLES / "flow.toml", overrides=overrides)
    assert list(training.dataset.groups) == list(cache.dataset.groups)
    for name, group in training.dataset.groups.items():
        assert group.resolution == cache.dataset.groups[name].resolution
        assert group.cache_directory == cache.dataset.groups[name].cache_directory
    assert json.dumps(overrides) == before
    with pytest.raises(ConfigError, match="share one calibrated resolution"):
        load_training_config(EXAMPLES / "restoration.toml", overrides=overrides)
