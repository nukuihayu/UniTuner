"""Exercise independent command boundaries without reading weights or media."""

from copy import deepcopy
from pathlib import Path

import pytest

from unitunner.cli.main import main
from unitunner.config.loading import ConfigError, load_cache_config, load_calibration_config

EXAMPLES = Path(__file__).resolve().parents[2] / "docs/examples"


def test_minimal_cache_and_calibration_need_no_training_arguments():
    cache = load_cache_config(EXAMPLES / "image-cache.toml")
    middle = load_calibration_config(EXAMPLES / "mid-timestep.toml")
    assert not hasattr(cache.model, "weights")
    assert not hasattr(middle.model, "text_encoder")
    assert not hasattr(middle.dataset.groups["main"], "cache_directory")
    assert main(["validate", "--for", "cache", str(EXAMPLES / "image-cache.toml")]) == 0
    assert main(["validate", "--for", "mid-timestep", str(EXAMPLES / "mid-timestep.toml")]) == 0


def test_restoration_cache_does_not_require_training_pair_sources():
    config = load_cache_config(
        EXAMPLES / "image-cache.toml",
        overrides={
            "dataset": {"groups": {"portraits": {"kind": "restoration"}}},
        },
    )
    assert config.dataset.groups["portraits"].kind == "restoration"
    assert not hasattr(config.dataset.groups["portraits"], "low_quality_source")


@pytest.mark.parametrize("command", ["cache", "mid-timestep"])
def test_shared_file_ignores_unrelated_training_validation(command):
    loader = load_cache_config if command == "cache" else load_calibration_config
    shared = EXAMPLES / "restoration.toml"
    config = loader(
        shared,
        overrides={
            "train": {"parameter_mode": "invalid", "adapter": None},
            "losses": {"gan": -2},
            "execution": {"precision": "invalid"},
            "method": {"restoration": {"dino_widths": []}},
        },
    )
    assert config.model.vae
    with pytest.raises(ConfigError):
        loader(shared, overrides={"model": {"vae": None}})
    with pytest.raises(ConfigError, match=r"extra_forbidden|Extra inputs"):
        loader(shared, overrides={"modle": {"vae": "typo"}})


@pytest.mark.parametrize("prompt", ["shared caption", ""])
def test_group_prompt_defaults_and_overrides(prompt):
    config = load_cache_config(
        EXAMPLES / "image-cache.toml",
        overrides={
            "dataset": {
                "defaults": {"prompt": prompt},
                "groups": {
                    "other": {"source": "other", "cache_directory": "other-cache", "prompt": "override"},
                },
            },
        },
    )
    assert config.dataset.groups["portraits"].prompt == prompt
    assert config.dataset.groups["other"].prompt == "override"
    values = deepcopy(config.model_dump(mode="json"))
    values["dataset"]["defaults"]["prompt"] = 12
    with pytest.raises(ConfigError):
        load_cache_config(overrides=values)


@pytest.mark.parametrize("command", ["cache", "mid-timestep"])
def test_help_excludes_training_options(command, capsys):
    with pytest.raises(SystemExit) as result:
        main([command, "--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--train.", "--losses.", "--execution.", "--model.weights"):
        assert option not in help_text
    if command == "mid-timestep":
        assert "--model.text-encoder" not in help_text
