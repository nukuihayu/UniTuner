"""Declare parameter selection, optimizer settings and update schedules."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from unituner.adapters.config import AdapterConfig
from unituner.config.diagnostics import ConfigRuleError
from unituner.config.optimizer import AdamWConfig


class TrainConfig(BaseModel):
    """Specify update limits independently of exported metadata."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    parameter_mode: Literal["adapter", "full"] = "adapter"
    adapter: AdapterConfig | None = None
    include_layers: tuple[str, ...] = Field(
        default=(), description="Native projection/module/parameter glob patterns; empty selects all eligible targets"
    )
    exclude_layers: tuple[str, ...] = Field(
        default=(), description="Native glob patterns removed after include_layers; exclusions take precedence"
    )
    learning_rate: float = Field(
        default=1e-4,
        ge=0,
        description="Base AdamW learning rate before schedule and parameter-group multipliers; 0 disables updates",
    )
    weight_decay: float = Field(default=0.01, ge=0, description="Decoupled AdamW weight decay; 0 disables decay")
    optimizer: AdamWConfig = Field(default_factory=AdamWConfig)
    max_updates: int = Field(default=1000, gt=0, description="Number of effective optimizer updates, not microbatches")
    accumulation_steps: int = Field(default=1, gt=0, description="Microbatches accumulated per optimizer update")
    max_grad_norm: float = Field(default=1.0, ge=0, description="Maximum global gradient norm; 0 disables clipping")
    seed: int = 42
    timestep_sampling: Literal["shifted", "uniform"] = Field(
        default="shifted",
        description="Flow training noise sampling: model-specific shifted timesteps or continuous uniform sigma",
    )
    skip_dataset_validation: bool = Field(
        default=False, description="Trust existing image caches and skip full dataset content validation at startup"
    )
    dataset_validation_workers: int = Field(
        default=4, ge=1, le=32, description="Parallel image dataset validation workers; 1 selects serial validation"
    )
    lr_schedule: Literal["constant", "linear", "cosine"] = "constant"
    warmup_updates: int = Field(
        default=0, ge=0, description="Effective updates for linear warmup; must be smaller than max_updates"
    )

    @field_validator("include_layers", "exclude_layers")
    @classmethod
    def validate_layer_patterns(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        """Reject blank patterns rather than silently selecting unexpected parameters."""
        if any(not pattern or pattern != pattern.strip() for pattern in patterns):
            raise ValueError("Layer patterns must be nonempty and have no surrounding whitespace")
        return patterns

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        """Keep warmup within the fixed effective-update horizon."""
        if self.warmup_updates >= self.max_updates:
            raise ConfigRuleError(
                "warmup_updates",
                "train.warmup_updates must be smaller than train.max_updates",
                "Reduce warmup_updates below max_updates, or increase max_updates.",
            )
        return self
