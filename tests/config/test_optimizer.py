"""Validate explicit AdamW policy without changing default continuation identity."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from unitunner.checkpoints.schema import build_resume_identity
from unitunner.config.loading import load_training_config
from unitunner.config.optimizer import AdamWConfig

EXAMPLE = Path(__file__).parents[2] / "docs/examples/flow.toml"


@pytest.mark.parametrize(
    "settings",
    [
        {"betas": [-0.1, 0.9]},
        {"betas": [0.9, 1.0]},
        {"betas": [0.9]},
        {"betas": [float("nan"), 0.9]},
        {"eps": 0.0},
        {"eps": float("inf")},
        {"backend": "unknown"},
        {"momentum": 0.9},
    ],
)
def test_invalid_optimizer_options_fail_at_configuration_boundary(settings):
    with pytest.raises(ValidationError):
        AdamWConfig.model_validate_json(json.dumps(settings))


def identity(config):
    return build_resume_identity(config, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64)


def test_explicit_defaults_share_current_grouped_optimizer_identity():
    baseline = load_training_config(EXAMPLE)
    explicit = load_training_config(
        EXAMPLE,
        overrides={
            "train": {
                "optimizer": {
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "backend": "auto",
                }
            }
        },
    )
    assert identity(baseline) == identity(explicit)
    assert identity(baseline).config_sha256 == "7d945df7cfbcb62bb218c8037a9052621b4104be6f22473c1ccecb4de289f447"


@pytest.mark.parametrize("settings", [{"betas": [0.8, 0.95]}, {"eps": 1e-6}, {"backend": "single"}])
def test_optimizer_overrides_are_pinned_in_resume_identity(settings):
    baseline = load_training_config(EXAMPLE)
    changed = load_training_config(EXAMPLE, overrides={"train": {"optimizer": settings}})
    assert identity(baseline) != identity(changed)


def test_omitted_and_disabled_optional_execution_flags_keep_identity():
    baseline = load_training_config(EXAMPLE)
    disabled = load_training_config(EXAMPLE, overrides={"execution": {"use_pinned_memory_for_block_swap": False}})
    assert baseline.execution.use_pinned_memory_for_block_swap is None
    assert identity(baseline) == identity(disabled)


def test_cache_batch_size_does_not_change_training_resume_identity():
    baseline = load_training_config(EXAMPLE)
    batched = load_training_config(EXAMPLE, overrides={"cache": {"batch_size": 16}})
    assert identity(baseline) == identity(batched)
