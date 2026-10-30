"""Declare local Wan video generation artifacts and optional high-noise expert."""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unituner.config.generation import GenerationAdapterConfig
from unituner.config.model import ModelConfig
from unituner.config.runtime import RuntimeConfig
from unituner.models.wan.config.sampling import WanSamplingConfig, validate_wan_sampling


class WanHighNoiseConfig(BaseModel):
    """Select a second matching2.2 expert, its CFG boundary and optional native adapter."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    weights: str = Field(min_length=1)
    boundary: float = Field(ge=0, le=1)
    guidance: float | None = Field(default=None, ge=0)
    adapter: GenerationAdapterConfig | None = None


class WanGenerationConfig(BaseModel):
    """Configure native FP32 video generation and an explicit MP4 or lossless MKV output."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig
    sampling: WanSamplingConfig = Field(default_factory=WanSamplingConfig)
    adapter: GenerationAdapterConfig | None = None
    high_noise: WanHighNoiseConfig | None = None
    output: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        """Validate the entire public contract before opening weights or media."""
        model = self.model
        if model.family != "wan" or model.vae is None or model.text_encoder is None:
            raise ValueError("Wan video generation requires Wan model, vae and text_encoder")
        validate_wan_sampling(self.sampling, version=model.version, variant=model.variant)
        if self.high_noise is not None and model.version != "2.2":
            raise ValueError("Wan dual expert generation requires version2.2")
        if Path(self.output).suffix.lower() not in (".mp4", ".mkv"):
            raise ValueError("Wan video output requires .mp4 or .mkv")
        return self
