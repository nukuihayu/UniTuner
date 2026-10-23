"""Declare native image-generation choices without importing compute backends."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unitunner.adapters.config import AdapterConfig
from unitunner.config.model import ModelConfig
from unitunner.config.runtime import RuntimeConfig
from unitunner.config.sampling import SamplingConfig, validate_image_sampling


class GenerationAdapterConfig(BaseModel):
    """Select one published-key adapter with explicit algorithm and rank."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    weights: str = Field(min_length=1)
    config: AdapterConfig


class GenerationConfig(BaseModel):
    """Select owned model artifacts, native sampling and a PNG output path."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    adapter: GenerationAdapterConfig | None = None
    output: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sampling(self) -> Self:
        """Validate native family controls without loading artifacts."""
        validate_image_sampling(
            self.sampling, family=self.model.family, variant=self.model.variant, version=self.model.version
        )
        return self
