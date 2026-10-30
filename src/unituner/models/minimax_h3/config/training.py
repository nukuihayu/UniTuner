"""Declare H3 local encoder artifacts and AV cache policies without loading models."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unituner.models.minimax_h3.config.audio_vae import MiniMaxH3AudioVAEConfig
from unituner.models.minimax_h3.config.video_vae import MiniMaxH3VideoVAEConfig


class MiniMaxH3EncoderConfig(BaseModel):
    """Select independent audio VAE, text config/processor and explicit source formats.

    model.VAE supplies video weights and model.text_encoder supplies text
    weights. External VAEs use published topology; native artifacts can embed custom topology.
    Text defaults to the command device unless text_device is explicitly set.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    audio_vae: str = Field(min_length=1)
    text_config: str = Field(min_length=1)
    text_processor: str | None = Field(default=None, min_length=1)
    text_device: Literal["cpu", "cuda"] | None = None
    video_topology: MiniMaxH3VideoVAEConfig | None = None
    audio_topology: MiniMaxH3AudioVAEConfig | None = None


class MiniMaxH3FlowConfig(BaseModel):
    """Validate dimensionless noise ranges, modality shifts and mean-loss weight."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)

    sigma_min: float = Field(default=0.0, ge=0, le=1)
    sigma_max: float = Field(default=1.0, ge=0, le=1)
    focus_min: float = Field(default=0.0, ge=0, le=1)
    focus_max: float = Field(default=1.0, ge=0, le=1)
    focus_probability: float = Field(default=0.0, ge=0, le=1)
    video_shift: float = Field(default=12.0, ge=0.01, le=100)
    audio_shift: float = Field(default=3.0, ge=0.01, le=100)
    visual_condition_clean: float = Field(default=0.999, ge=0, le=1)
    audio_condition_clean: float = Field(default=1.0, ge=0, le=1)
    audio_loss_weight: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        """Reject reversed or empty sampling bands before tensor execution."""
        if self.sigma_min >= self.sigma_max or self.focus_min >= self.focus_max:
            raise ValueError("H3 sigma and focus ranges must have increasing endpoints")
        if self.focus_probability > 0 and not self.sigma_min <= self.focus_min < self.focus_max <= self.sigma_max:
            raise ValueError("H3 active focus band must lie inside the base sigma range")
        return self
