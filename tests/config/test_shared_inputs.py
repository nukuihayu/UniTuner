"""Check common input semantics and native capability boundaries without model IO."""

import pytest
from pydantic import ValidationError

from unitunner.config.dataset_inputs import DatasetInputConfig
from unitunner.config.datasets import DatasetConfig
from unitunner.config.text import TextEncodingConfig


def test_ordered_references_keep_native_geometry_and_reject_unsupported_overrides():
    references = DatasetInputConfig(reference_sources=("first", "second", "first"))
    references.validate_for("flux2", "base")
    references.validate_for("qwen-image", "edit-2511")
    assert references.reference_sources == ("first", "second", "first")
    with pytest.raises(ValueError, match="does not support"):
        references.validate_for("z-image", "base")
    with pytest.raises(ValueError, match="exactly one"):
        references.validate_for("flux", "kontext")
    with pytest.raises(ValueError, match="at most ten"):
        DatasetInputConfig(reference_sources=("ref",) * 11).validate_for("flux3", "base")
    with pytest.raises(ValidationError, match="Extra inputs"):
        DatasetConfig.model_validate(
            {"source": "images", "cache_directory": "cache", "flux2": {"reference_sources": ("ref",)}}
        )


def test_text_overrides_preserve_native_budgets_and_feature_capabilities():
    native = TextEncodingConfig()
    assert native.resolved_max_length("moss-image") == 4096
    assert native.resolved_max_length("z-image") == 512
    text = TextEncodingConfig(max_length=256)
    for family in ("z-image", "moss-image", "flux", "krea2"):
        text.validate_for(family)
        assert text.resolved_max_length(family) == 256
    with pytest.raises(ValueError, match="fixed native text budget"):
        text.validate_for("flux3")
    with pytest.raises(ValidationError, match="Extra inputs"):
        TextEncodingConfig.model_validate({"select_layers": (2, 0, 1)})
