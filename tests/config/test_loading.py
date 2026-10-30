"""Check actual strict boundary behavior, method defaults and override precedence."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import JsonValue, ValidationError

from tests.support.assertions import require_value
from unitunner.cli.main import main
from unitunner.config.loading import ConfigError, load_cache_config, load_calibration_config, load_training_config

EXAMPLES = Path(__file__).resolve().parents[2] / "docs/examples"


def test_cache_batch_size_is_independent_of_training_and_calibration():
    path = EXAMPLES / "restoration.toml"
    overrides = {"cache": {"batch_size": 16}}
    cache = load_cache_config(path, overrides=overrides)
    training = load_training_config(path, overrides=overrides)
    assert cache.cache.batch_size == training.cache.batch_size == 16
    assert training.dataset.groups["main"].batch_size == 1
    assert load_calibration_config(path, overrides=overrides).calibration_pairs > 0
    for value in (0, -1, True, 1.5):
        with pytest.raises(ConfigError, match=r"cache\.batch_size"):
            load_cache_config(path, overrides={"cache": {"batch_size": value}})
    with pytest.raises(ConfigError, match="Z-Image only"):
        load_cache_config(EXAMPLES / "flux-restoration.toml", overrides=overrides)


def test_cache_batch_size_cli_and_shared_training_config(monkeypatch):
    received = []

    def run(config, *, device):
        received.append((config.cache.batch_size, device))
        return 0

    monkeypatch.setattr("unitunner.cli.main.import_module", lambda name: SimpleNamespace(run=run))
    assert main(["cache", str(EXAMPLES / "restoration.toml"), "--cache.batch-size", "16", "--device", "cpu"]) == 0
    assert received == [(16, "cpu")]


@pytest.mark.parametrize("loader", [load_training_config, load_cache_config, load_calibration_config])
def test_vae_field_is_shared_and_old_codec_field_is_rejected(loader):
    config = loader(EXAMPLES / "restoration.toml", overrides={"model": {"vae": "models/vae.safetensors"}})
    assert config.model.vae == "models/vae.safetensors"
    with pytest.raises(ConfigError, match=r"model.codec"):
        loader(EXAMPLES / "restoration.toml", overrides={"model": {"codec": "models/old.safetensors"}})


@pytest.mark.parametrize("name", ["flow", "full", "restoration"])
def test_examples(name):
    config = load_training_config(EXAMPLES / f"{name}.toml")
    assert config.train.max_updates > 0
    if name == "restoration":
        losses = config.losses
        assert losses is not None
        assert (losses.lrr, losses.dv3d, losses.l1, losses.gan, losses.fdl, losses.histogram) == (
            5.0,
            3.0,
            0.5,
            0.5,
            0.5,
            0.0,
        )


def test_override_is_explicit_and_nested():
    config = load_training_config(EXAMPLES / "flow.toml", overrides={"train": {"learning_rate": 0.003}})
    assert config.train.learning_rate == 0.003
    assert config.train.seed == 42
    with pytest.raises(ValidationError, match="frozen"):
        config.train.seed = 1


def test_optimizer_decay_default_and_restoration_recipe():
    config = load_training_config(EXAMPLES / "restoration.toml")
    assert config.train.learning_rate == 5e-5
    assert config.train.weight_decay == 0.01
    config = load_training_config(EXAMPLES / "flow.toml")
    assert config.train.weight_decay == 0.01
    config = load_training_config(EXAMPLES / "restoration.toml", overrides={"train": {"weight_decay": 0.0}})
    assert config.train.weight_decay == 0.0


def test_restoration_midtimestep_cli_overrides_toml_before_validation(monkeypatch):
    """Forward a validated nondefault time to training; reject ignored or invalid choices."""
    received = []

    def run(config, *, device, resume):
        received.append(config.method.restoration.midtimestep)
        return 0

    monkeypatch.setattr("unitunner.cli.main.import_module", lambda name: SimpleNamespace(run=run))
    path = str(EXAMPLES / "restoration.toml")
    assert require_value(load_training_config(Path(path)).method.restoration).midtimestep == 244
    assert main(["train", path, "--midtimestep", "321"]) == 0
    assert received == [321]
    assert main(["train", path, "--midtimestep", "1000"]) == 2
    assert main(["train", str(EXAMPLES / "flow.toml"), "--midtimestep", "321"]) == 2
    for options in ({"midtimestep": 321, "sigma": 0.3}, {"midtimestep": 321, "calibrate": True}):
        config = load_training_config(Path(path), overrides={"method": {"restoration": options}})
        assert config.method.restoration is not None
        assert config.method.restoration.midtimestep == 321


@pytest.mark.parametrize(
    "override",
    [
        {"train": {"learning_arte": 0.1}},
        {"train": {"learning_rate": "0.1"}},
        {"train": {"accumulation_steps": True}},
        {"train": {"max_updates": 0}},
        {"train": {"learning_rate": float("nan")}},
        {"train": {"weight_decay": -0.01}},
        {"train": {"weight_decay": float("nan")}},
        {"train": {"weight_decay": float("inf")}},
        {"dataset": {"groups": {"main": {"resolution": [0, 10]}}}},
        {"training": {"learning_rate": 0.1}},
        {"full_fp16": True},
        {"model": {"family": "wan"}, "train": {"parameter_mode": "full"}},
        {"train": {"parameter_mode": "full"}, "execution": {"quantization": "fp8"}},
        {"method": {"kind": "restoration"}},
        {"method": {"restoration": {"gan": 0.0, "dv3d": 0.0, "fdl": 0.0}}},
    ],
)
def test_invalid_input_rejected_before_io(override):
    with pytest.raises(ConfigError):
        load_training_config(EXAMPLES / "flow.toml", overrides=override)


def test_diagnostics_do_not_echo_secret():
    with pytest.raises(ConfigError) as error:
        load_training_config(EXAMPLES / "flow.toml", overrides={"train": {"learning_rate": "secret-password"}})
    assert "secret-password" not in str(error.value)
    assert "train.learning_rate" in str(error.value)


def test_calibration_requires_unambiguous_restoration_configuration():
    config = load_training_config(
        EXAMPLES / "restoration.toml",
        overrides={"method": {"restoration": {"calibrate": True, "calibration_pairs": 2}}},
    )
    assert config.method.restoration is not None
    assert config.method.restoration.calibration_pairs == 2
    invalid_options: tuple[dict[str, JsonValue], ...] = (
        {"calibrate": True, "sigma": 0.3},
        {"calibration_pairs": 2},
        {"calibrate": True, "calibration_pairs": 0},
    )
    for options in invalid_options:
        with pytest.raises(ConfigError):
            load_training_config(EXAMPLES / "restoration.toml", overrides={"method": {"restoration": options}})


def test_synthetic_degradation_schema_rejects_ambiguous_or_invalid_distributions():
    base = {"dataset": {"groups": {"main": {"low_quality_source": None, "degradation": {"enabled": True}}}}}
    config = load_training_config(EXAMPLES / "restoration.toml", overrides=base)
    assert config.dataset.groups["main"].degradation is not None
    assert config.dataset.groups["main"].degradation.enabled
    encoded = config.model_dump_json()
    assert type(config).model_validate_json(encoded) == config
    overrides = [
        {"dataset": {"groups": {"main": {"degradation": {"enabled": True}}}}},
        {"dataset": {"groups": {"main": {"low_quality_source": None}}}},
        {"dataset": {"groups": {"main": {"low_quality_source": None, "degradation": {"enabled": True, "scales": []}}}}},
        {
            "dataset": {
                "groups": {
                    "main": {
                        "low_quality_source": None,
                        "degradation": {"enabled": True, "first": {"resize_probabilities": [0.5, 0.5, 0.5]}},
                    }
                }
            }
        },
        {
            "dataset": {
                "groups": {
                    "main": {"low_quality_source": None, "degradation": {"enabled": True}, "resolution": [16, 32]}
                }
            }
        },
    ]
    overrides.append(
        {
            "dataset": {
                "groups": {
                    "main": {
                        "low_quality_source": None,
                        "degradation": {"enabled": True, "first": {"noise": {"probabilities": [0.3, 0.2, 0.5000005]}}},
                    }
                }
            }
        }
    )
    for override in overrides:
        with pytest.raises(ConfigError):
            load_training_config(EXAMPLES / "restoration.toml", overrides=override)


def test_named_groups_require_distinct_cache_roles():
    groups = {
        "paired": {
            "kind": "restoration",
            "source": "hq-a",
            "low_quality_source": "lq-a",
            "cache_directory": "cache-a",
            "repeats": 2,
        },
        "synthetic": {
            "kind": "restoration",
            "source": "hq-b",
            "cache_directory": "cache-b",
            "degradation": {"enabled": True},
        },
    }
    config = load_training_config(EXAMPLES / "restoration-mixed.toml", overrides={"dataset": {"groups": groups}})
    assert "paired" in config.dataset.groups
    assert type(config).model_validate_json(config.model_dump_json()) == config
    for invalid in (
        {"dataset": {"groups": {"synthetic": {"cache_directory": "/data/cache-aligned"}}}},
        {"dataset": {"groups": {"synthetic": {"degradation": None}}}},
    ):
        with pytest.raises(ConfigError):
            load_training_config(EXAMPLES / "restoration-mixed.toml", overrides=invalid)


def test_preview_configuration_requires_matching_inputs():
    valid = {"every_updates": 2, "samples": [{"prompt": "preview", "width": 32, "height": 16}]}
    config = load_training_config(EXAMPLES / "flow.toml", overrides={"preview": valid})
    assert config.preview is not None
    assert config.preview.every_updates == 2
    for preview in ({}, {**valid, "num_images": 2}, {**valid, "every_updates": 0}):
        with pytest.raises(ConfigError):
            load_training_config(EXAMPLES / "flow.toml", overrides={"preview": preview})
    with pytest.raises(ConfigError):
        load_training_config(EXAMPLES / "restoration.toml", overrides={"preview": valid})


def test_restoration_wavelet_objective_needs_no_feature_models():
    losses = dict.fromkeys(("lrr", "l1", "dv3d", "gan", "fdl", "histogram", "phase"), 0.0)
    method = {"restoration": {"dino_weights": None, "vgg_weights": None}, "wavelet": {"wavelet": "haar"}}
    config = load_training_config(EXAMPLES / "restoration.toml", overrides={"losses": losses, "method": method})
    assert config.method.wavelet is not None and config.method.wavelet.wavelet == "haar"


@pytest.mark.parametrize("bands", [{"ll": 0.0}, {"ll": 1.0, "ll_level_threshold": 3}])
def test_restoration_rejects_ineffective_wavelet_only_objective(bands):
    losses = dict.fromkeys(("lrr", "l1", "dv3d", "gan", "fdl", "histogram", "phase"), 0.0)
    with pytest.raises(ConfigError, match="At least one image objective"):
        load_training_config(
            EXAMPLES / "restoration.toml",
            overrides={
                "losses": losses,
                "method": {"wavelet": {"levels": 3, "lh": 0.0, "hl": 0.0, "hh": 0.0, **bands}},
            },
        )
