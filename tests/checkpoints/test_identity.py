"""Ensure output metadata cannot change mathematical identity or update limits."""

import json
from pathlib import Path

import pytest

from unitunner.checkpoints.metadata import Provenance
from unitunner.checkpoints.schema import build_resume_identity
from unitunner.config.loading import load_training_config

EXAMPLE = Path(__file__).resolve().parents[2] / "docs/examples/flow.toml"


def test_resume_identity_pins_math_not_output():
    config = load_training_config(EXAMPLE)
    output = load_training_config(EXAMPLE, overrides={"output": {"directory": "elsewhere", "save_every_updates": 5}})
    changed = load_training_config(EXAMPLE, overrides={"train": {"accumulation_steps": 2}})

    def identity(value):
        return build_resume_identity(value, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64)

    preview = load_training_config(EXAMPLE, overrides={"preview": {"samples": [{"prompt": "preview"}]}})
    assert identity(config) == identity(preview)
    assert identity(config) == identity(output)
    assert identity(config) != identity(changed)
    decay = load_training_config(EXAMPLE, overrides={"train": {"weight_decay": 0.01}})
    assert identity(config) != identity(decay)


def test_dataset_validation_policy_does_not_change_resume_identity():
    config = load_training_config(EXAMPLE)
    trusted = load_training_config(
        EXAMPLE, overrides={"train": {"skip_dataset_validation": True, "dataset_validation_workers": 8}}
    )
    assert not config.train.skip_dataset_validation
    assert config.train.dataset_validation_workers == 4
    assert trusted.train.skip_dataset_validation
    assert build_resume_identity(config, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64) == (
        build_resume_identity(trusted, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64)
    )


@pytest.mark.parametrize(
    "example,policy",
    [
        ("qwen-image-flow.toml", "qwen/native-clock-cast/divide1000/cast/1"),
        ("qwen-image-2-1-flow.toml", "qwen21/native-clock-cast/divide1000/cast/1"),
        ("flux3-flow.toml", "flux3/fraction-cast/multiply1000/1"),
    ],
)
@pytest.mark.parametrize("precision", ["bf16", "fp16", "fp32"])
def test_timestep_rounding_policy_is_pinned_only_for_low_precision(monkeypatch, example, policy, precision):
    from unitunner.checkpoints import schema

    config = load_training_config(EXAMPLE.with_name(example), overrides={"execution": {"precision": precision}})
    captured = {}
    original = schema.canonical_sha256

    def capture(values):
        captured.update(values)
        return original(values)

    monkeypatch.setattr(schema, "canonical_sha256", capture)
    build_resume_identity(config, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64)
    assert captured.get("timestep_precision_policy") == (None if precision == "fp32" else policy)


def test_provenance_needs_no_deleted_cli_attributes():
    provenance = Provenance(
        model_family="z-image",
        variant="base",
        method="flow-matching",
        parameter_mode="full",
        precision="bf16",
        completed_updates=3,
        config_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        source_sha256="c" * 64,
    )
    metadata = provenance.metadata()
    assert all(isinstance(value, str) for value in metadata.values())
    assert json.loads(metadata["unitunner.provenance"])["completed_updates"] == 3


def test_execution_defaults_pin_image_and_wan_numerical_identities():
    """Pin compiler and dataset policies that determine numerical continuation."""
    published = {
        "flow.toml": "4a25dbfefe8fcbd2d79373c4bff09086cdce16fff09be0605866bb47bb5f314c",
        "wan-flow.toml": "fa26924e6c17fe34393d041315943ab18546078347fb6f3ee1904899227f6aff",
        "restoration.toml": "4d019406e108b2e8bb73d76228dd143c4f7b6fdf5364e91defd63d9dc5a159f4",
    }
    for name, expected in published.items():
        config = load_training_config(EXAMPLE.with_name(name))
        identity = build_resume_identity(config, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64)
        assert identity.config_sha256 == expected


def test_ffn_recompute_configuration_is_rejected():
    """Removed execution policies must not silently select a different memory strategy."""
    from pydantic import ValidationError

    from unitunner.config.execution import ExecutionConfig

    with pytest.raises(ValidationError, match="recompute"):
        ExecutionConfig.model_validate({"recompute": "ffn"})


def test_inactive_quantization_geometry_does_not_change_resume_identity():
    baseline = load_training_config(EXAMPLE)
    geometry = load_training_config(
        EXAMPLE, overrides={"execution": {"quantization_rotation_size": 64, "quantization_group_size": 32}}
    )

    def identity(config):
        return build_resume_identity(config, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64)

    assert identity(baseline) == identity(geometry)
    quantized = load_training_config(EXAMPLE, overrides={"execution": {"quantization": "int8_convrot"}})
    quantized_geometry = load_training_config(
        EXAMPLE,
        overrides={
            "execution": {
                "quantization": "int8_convrot",
                "quantization_rotation_size": 64,
                "quantization_group_size": 32,
            }
        },
    )
    assert identity(quantized) != identity(quantized_geometry)


def test_phase_identity_preserves_disabled_runs_and_pins_enabled_parameters():
    example = EXAMPLE.with_name("restoration.toml")

    def identity(weight=0.0, **options):
        config = load_training_config(
            example, overrides={"losses": {"phase": weight}, "method": {"restoration": options}}
        )
        return build_resume_identity(config, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64)

    assert identity() == identity(phase_period=8, phase_cycles=4, phase_epsilon=0.01)
    enabled = identity(4.0)
    assert enabled != identity()
    assert enabled != identity(2.0)
    for options in ({"phase_period": 8}, {"phase_cycles": 4}, {"phase_epsilon": 0.01}):
        assert enabled != identity(4.0, **options)
