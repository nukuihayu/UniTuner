"""Declare only VAE, paired images and numerical controls for SNR calibration."""

import json
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationInfo, field_validator, model_validator

from unitunner.config.datasets import DatasetConfig
from unitunner.config.diagnostics import ConfigRuleError
from unitunner.config.model import EncoderModelConfig
from unitunner.config.runtime import RuntimeConfig
from unitunner.data.degradation.config import DegradationConfig
from unitunner.models.catalog import ModelFamily, supports_restoration


class CalibrationModelConfig(BaseModel):
    """Calibration never constructs denoiser or text encoder components."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    family: ModelFamily
    vae: str = Field(min_length=1)


class CalibrationDefaults(BaseModel):
    """Inherit image geometry and augmentation policy independently of batching."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    resolution: tuple[int, int] = (1024, 1024)
    repeats: int = Field(default=1, gt=0)
    horizontal_flip_probability: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)


class CalibrationDatasetConfig(CalibrationDefaults):
    """Require paired images or a synthetic degradation recipe; no cache path."""

    source: str = Field(min_length=1)
    low_quality_source: str | None = None
    degradation: DegradationConfig | None = None

    @model_validator(mode="after")
    def validate_pairs(self) -> Self:
        synthetic = self.degradation is not None and self.degradation.enabled
        if synthetic == bool(self.low_quality_source):
            raise ConfigRuleError(
                "low_quality_source",
                "Calibration requires either low_quality_source or enabled degradation",
                "Supply aligned LQ images or enable synthetic degradation, but do not select both.",
            )
        if synthetic and min(self.resolution) < 32:
            raise ConfigRuleError(
                "resolution",
                "Synthetic degradation requires resolution at least 32",
                "Set both dimensions to at least 32 for synthetic degradation.",
            )
        if any(size < 16 or size % 16 for size in self.resolution):
            raise ConfigRuleError(
                "resolution",
                "Calibration resolution requires 16-aligned dimensions",
                "Set both dimensions to positive multiples of 16, e.g. [512,512].",
            )
        return self


class CalibrationDataConfig(BaseModel):
    """Resolve independent source groups without requiring training cache metadata."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    defaults: CalibrationDefaults = Field(default_factory=CalibrationDefaults)
    groups: dict[str, CalibrationDatasetConfig] = Field(min_length=1)

    @field_validator("groups", mode="before")
    @classmethod
    def inherit_defaults(cls, groups: Any, info: ValidationInfo) -> Any:
        defaults = info.data.get("defaults")
        if not isinstance(groups, dict) or not isinstance(defaults, CalibrationDefaults):
            return groups
        merged = {
            name: {**defaults.model_dump(mode="json"), **group} if isinstance(group, dict) else group
            for name, group in groups.items()
        }
        if all(isinstance(group, CalibrationDatasetConfig) for group in merged.values()):
            return merged
        return TypeAdapter(dict[str, CalibrationDatasetConfig]).validate_json(json.dumps(merged), strict=True)


class CalibrationConfig(BaseModel):
    """Configure a read-only VAE calibration without validating training settings."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: CalibrationModelConfig
    dataset: CalibrationDataConfig
    seed: int = 42
    posterior: Literal["mean", "sample"] | None = None
    calibration_pairs: int = Field(default=32, gt=0)
    calibration_sampling: Literal["random", "tiles"] = "random"

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        if not supports_restoration(self.model.family):
            raise ValueError("Selected family does not support image restoration calibration")
        if len({group.resolution for group in self.dataset.groups.values()}) != 1:
            raise ValueError("Calibration groups must share a resolution")
        if any(not name.isidentifier() or not name.isascii() for name in self.dataset.groups):
            raise ValueError("Calibration group names must be ASCII identifiers")
        return self


def calibration_document(document: dict[str, Any]) -> dict[str, Any]:
    """Project known shared run fields; unknown keys remain validation errors."""
    result = dict(document)
    train = result.pop("train", {})
    method = result.pop("method", {})
    for section in ("losses", "execution", "output", "preview"):
        result.pop(section, None)
    if isinstance(train, dict) and "seed" in train:
        result.setdefault("seed", train["seed"])
    options = method.get("restoration", {}) if isinstance(method, dict) else {}
    if isinstance(options, dict):
        for field in ("posterior", "calibration_pairs", "calibration_sampling"):
            if field in options:
                result.setdefault(field, options[field])
    model = result.get("model")
    if isinstance(model, dict):
        unused = (EncoderModelConfig.model_fields.keys() | {"weights"}) - CalibrationModelConfig.model_fields.keys()
        result["model"] = {key: value for key, value in model.items() if key not in unused}
    data = result.get("dataset")
    if isinstance(data, dict):
        data = dict(data)
        unused = DatasetConfig.model_fields.keys() - CalibrationDatasetConfig.model_fields.keys()
        for key in ("defaults", "groups"):
            section = data.get(key)
            if isinstance(section, dict):
                if key == "defaults":
                    data[key] = {name: value for name, value in section.items() if name not in unused}
                else:
                    data[key] = {
                        name: {field: value for field, value in group.items() if field not in unused}
                        if isinstance(group, dict)
                        else group
                        for name, group in section.items()
                    }
        result["dataset"] = data
    return result
