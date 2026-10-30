"""Protect validation boundaries and the stable serialized configuration schema."""

import pytest

from tests.config.test_data_groups import document
from unitunner.config.execution import ExecutionConfig
from unitunner.config.loading import load_training_config
from unitunner.config.output import OutputConfig
from unitunner.config.train import TrainConfig
from unitunner.config.training import OutputConfig as PublishedOutputConfig
from unitunner.config.training import TrainConfig as PublishedTrainConfig


def test_global_validation_runs_once_and_group_checks_retain_each_group(monkeypatch):
    from unitunner.config import training

    counts = []
    for name in ("validate_training_policy", "validate_training_preview", "validate_training_dataset"):
        original = getattr(training, name)

        def record(*args, _original=original, _name=name):
            counts.append(_name)
            return _original(*args)

        monkeypatch.setattr(training, name, record)
    config = load_training_config(overrides=document())
    assert len(config.dataset.groups) == 2
    assert counts.count("validate_training_policy") == 1
    assert counts.count("validate_training_preview") == 1
    assert counts.count("validate_training_dataset") == 2


@pytest.mark.parametrize(
    "residency, offloads", [("resident", False), ("frozen-blocks", True), ("vbar", True), ("trainable-blocks", False)]
)
def test_derived_residency_policy_does_not_change_serialized_config(residency, offloads):
    config = ExecutionConfig(residency=residency)
    assert config.offloads_frozen_weights is offloads
    assert "offloads_frozen_weights" not in config.model_dump()
    assert config.model_validate_json(config.model_dump_json()) == config


def test_config_classes_remain_importable_and_output_identity_survives_roundtrip():
    assert PublishedOutputConfig is OutputConfig
    assert PublishedTrainConfig is TrainConfig
    config = load_training_config(overrides=document())
    config.output._run_name = "named-run"
    assert "_run_name" not in config.output.model_dump()
    assert config.model_validate_json(config.model_dump_json()).model_dump() == config.model_dump()


def test_adapter_format_default_override_and_algorithm_validation():
    assert OutputConfig().adapter_format == "comfyui"
    value = document()
    value["output"] = {"adapter_format": "diffusers"}
    config = load_training_config(overrides=value)
    assert config.output.adapter_format == "diffusers"
    for algorithm in ("loha", "lokr"):
        value["train"]["adapter"] = {"algorithm": algorithm}
        with pytest.raises(ValueError, match="vanilla LoRA"):
            load_training_config(overrides=value)
