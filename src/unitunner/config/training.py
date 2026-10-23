"""Assemble immutable training configuration and validate cross-component contracts."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unitunner.config.cache_options import CacheOptions
from unitunner.config.datasets import DataConfig
from unitunner.config.diagnostics import ConfigRuleError
from unitunner.config.execution import ExecutionConfig
from unitunner.config.model import ModelConfig as ModelConfig
from unitunner.config.output import OutputConfig as OutputConfig
from unitunner.config.previews import PreviewConfig
from unitunner.config.restoration import MethodConfig, RestorationLossWeights
from unitunner.config.runtime import RuntimeConfig
from unitunner.config.train import TrainConfig as TrainConfig
from unitunner.config.training_validation import (
    validate_training_dataset,
    validate_training_policy,
    validate_training_preview,
)


class TrainingConfig(BaseModel):
    """Own immutable user choices validated once at the input boundary."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig
    dataset: DataConfig
    cache: CacheOptions = Field(default_factory=CacheOptions)
    train: TrainConfig = Field(default_factory=TrainConfig)
    method: MethodConfig = Field(default_factory=MethodConfig)
    losses: RestorationLossWeights | None = None
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    preview: PreviewConfig | None = None

    @model_validator(mode="after")
    def validate_combinations(self) -> Self:
        """Validate run policy once, then attach group names to dataset errors."""
        if self.method.kind == "restoration" and len({group.resolution for group in self.dataset.groups.values()}) != 1:
            raise ConfigRuleError(
                "dataset.groups",
                "Restoration data groups must share one calibrated resolution",
                "Set every restoration group to the same resolution used for calibration.",
            )
        validate_training_policy(self)
        validate_training_preview(self)
        for name, dataset in self.dataset.groups.items():
            try:
                validate_training_dataset(self, dataset)
            except ValueError as error:
                field = f"dataset.groups.{name}"
                hint = "Check this data group against the selected model family, variant and method."
                if isinstance(error, ConfigRuleError):
                    field += f".{error.field}" if error.field else ""
                    hint = error.hint
                raise ConfigRuleError(field, str(error), hint) from None
        return self
