"""Accept timestep sampling types without numeric shift overrides."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from unitunner.config.loading import load_training_config
from unitunner.config.train import TrainConfig


def test_model_shifted_sampling_is_default_and_uniform_is_selectable():
    assert TrainConfig().timestep_sampling == "shifted"
    assert TrainConfig(timestep_sampling="uniform").timestep_sampling == "uniform"


@pytest.mark.parametrize("value", [0, 1, 3.0, True, "auto", "logit-normal", None])
def test_numeric_and_unknown_sampling_choices_are_rejected(value):
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"timestep_sampling": value})


def test_shift_multiplier_is_not_a_config_option():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"timestep_shift": 3.0})


def test_restoration_does_not_accept_flow_sampling_override():
    example = Path(__file__).parents[2] / "docs/examples/restoration.toml"
    config = load_training_config(example)
    assert config.train.timestep_sampling == "shifted"
    with pytest.raises(ValueError, match="timestep_sampling"):
        load_training_config(example, overrides={"train": {"timestep_sampling": "uniform"}})
