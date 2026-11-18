"""Exercise user-facing configuration diagnostics across command boundaries."""

import argparse
from datetime import date
from pathlib import Path

import pytest

from tests.config.test_data_groups import document
from unituner.cli.configuration import add_config_arguments
from unituner.cli.main import main
from unituner.config.diagnostics import ConfigError
from unituner.config.loading import (
    load_cache_config,
    load_calibration_config,
    load_generation_config,
    load_restoration_inference_config,
    load_training_config,
    load_video_generation_config,
)
from unituner.config.train import TrainConfig

EXAMPLES = Path(__file__).parents[2] / "docs/examples"


def failure(overrides, example="flow.toml"):
    with pytest.raises(ConfigError) as error:
        load_training_config(EXAMPLES / example, overrides=overrides)
    return error.value


def test_reports_all_schema_errors_with_types_ranges_sources_and_typo_suggestions():
    error = failure({"train": {"max_updates": 0, "learning_rate": "secret-password", "learning_arte": 0.1}})
    assert len(error.issues) == 3
    issues = {issue.field: issue for issue in error.issues}
    assert issues["train.max_updates"].received == "0"
    assert issues["train.max_updates"].expected == "integer; > 0"
    assert "effective optimizer updates" in issues["train.max_updates"].description
    assert "learning_rate" in issues["train.learning_arte"].hint
    assert issues["train.learning_rate"].source == "CLI override"
    assert "quoted numbers" in issues["train.learning_rate"].hint
    assert "secret-password" not in str(error)
    assert "3 configuration error(s)" in str(error)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_keeps_field_location(number):
    error = failure({"train": {"learning_rate": number}})
    assert error.issues[0].field == "train.learning_rate"
    assert "finite" in error.issues[0].hint


def test_unsupported_date_points_to_field_instead_of_generic_serialization_error():
    error = failure({"train": {"seed": date(2026, 10, 22)}})
    assert error.issues[0].field == "train.seed"
    assert error.issues[0].received == "date"
    assert "quote dates" in error.issues[0].hint


def test_cross_field_rules_identify_leaf_and_remedy():
    error = failure({"train": {"warmup_updates": 1000}})
    issue = error.issues[0]
    assert issue.field == "train.warmup_updates"
    assert issue.received == "1000"
    assert "max_updates" in issue.hint
    error = failure({"method": {"restoration": {"dino_weights": None}}}, "restoration.toml")
    assert error.issues[0].field == "method.restoration.dino_weights"
    assert "losses.dv3d=0" in error.issues[0].hint
    error = failure({"method": {"restoration": {"calibrate": True, "sigma": 0.3}}}, "restoration.toml")
    assert error.issues[0].field == "method.restoration.sigma"
    assert "calibrate=false" in error.issues[0].hint


def test_dataset_rule_and_unknown_key_preserve_named_group():
    values = document()
    values["dataset"]["groups"]["landscapes"]["batch_siz"] = 2
    with pytest.raises(ConfigError) as caught:
        load_training_config(overrides=values)
    issue = caught.value.issues[0]
    assert issue.field == "dataset.groups.landscapes.batch_siz"
    assert "batch_size" in issue.hint
    values["dataset"]["groups"]["landscapes"].pop("batch_siz")
    values["dataset"]["groups"]["landscapes"]["resolution"] = [0, 10]
    with pytest.raises(ConfigError) as caught:
        load_training_config(overrides=values)
    assert caught.value.issues[0].field == "dataset.groups.landscapes.resolution"
    assert "positive" in caught.value.issues[0].hint


def test_inherited_defaults_have_correct_input_origin():
    values = document()
    for group in values["dataset"]["groups"].values():
        group.pop("resolution", None)
    values["dataset"]["defaults"]["resolution"] = [0, 512]
    with pytest.raises(ConfigError) as caught:
        load_training_config(overrides=values)
    assert all("dataset.groups." in issue.field for issue in caught.value.issues)
    assert all("CLI override (inherited from dataset.defaults)" == issue.source for issue in caught.value.issues)


def test_union_branch_labels_do_not_appear_as_configuration_keys():
    error = failure({"execution": {"compile": "secret-value"}})
    assert all(issue.field == "execution.compile" for issue in error.issues)
    assert all("boolean" in issue.expected and "auto" in issue.expected for issue in error.issues)
    assert "secret-value" not in str(error)


def test_toml_syntax_and_unreadable_file_errors_are_actionable(tmp_path):
    path = tmp_path / "broken.toml"
    path.write_text("[train]\nmax_updates = [1,\n", encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_training_config(path)
    assert "invalid TOML" in str(caught.value)
    assert "line" in str(caught.value) or "end of document" in str(caught.value)
    assert "quoting" in str(caught.value)
    with pytest.raises(ConfigError, match="cannot read configuration"):
        load_training_config(tmp_path / "missing.toml")
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_training_config(path)


@pytest.mark.parametrize(
    "loader",
    [
        load_training_config,
        load_cache_config,
        load_calibration_config,
        load_generation_config,
        load_restoration_inference_config,
        load_video_generation_config,
    ],
)
def test_every_loader_uses_structured_errors(loader):
    with pytest.raises(ConfigError) as caught:
        loader(overrides={})
    assert caught.value.issues
    assert all(issue.code == "missing" for issue in caught.value.issues)
    assert "--set" in str(caught.value)


def test_cli_help_uses_schema_descriptions_constraints_and_safe_percent_literals():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, TrainConfig)
    help_text = parser.format_help()
    assert "Effective updates" in help_text or "effective optimizer updates" in help_text
    assert "Expected: integer; > 0" in help_text


def test_invalid_cli_configuration_stops_before_runtime_import(monkeypatch, capsys):
    def unexpected_import(name):
        pytest.fail(f"runtime imported for invalid config: {name}")

    monkeypatch.setattr("unituner.cli.main.import_module", unexpected_import)
    assert main(["train", str(EXAMPLES / "flow.toml"), "--train.max-updates", "0"]) == 2
    stderr = capsys.readouterr().err
    assert "train.max_updates" in stderr
    assert "received: 0" in stderr
    assert "Expected: integer; > 0" in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize(
    "overrides, example, field, received",
    [
        ({"train": {"optimizer": {"betas": [0.9, 1.0]}}}, "flow.toml", "train.optimizer.betas.1", "1.0"),
        (
            {"method": {"restoration": {"dino_widths": [192, 384, 0, 1536]}}},
            "restoration.toml",
            "method.restoration.dino_widths.2",
            "0",
        ),
    ],
)
def test_numeric_array_errors_identify_element_and_value(overrides, example, field, received):
    issue = failure(overrides, example).issues[0]
    assert issue.field == field
    assert issue.received == received
    assert issue.expected


def test_file_values_have_toml_origin(tmp_path):
    path = tmp_path / "config.toml"
    content = (EXAMPLES / "flow.toml").read_text(encoding="utf-8")
    assert "eps = 1e-08" in content
    path.write_text(content.replace("eps = 1e-08", "eps = 0"), encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_training_config(path)
    assert caught.value.issues[0].source == "TOML/default"
    assert caught.value.issues[0].field == "train.optimizer.eps"
    assert str(path) in str(caught.value)
