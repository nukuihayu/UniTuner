"""Validate shared audio decode clocks, channel geometry and repair limits."""

from pydantic import BaseModel, ConfigDict, Field


class AudioDecodeConfig(BaseModel):
    """Define a resampled waveform clock and maximum repairable PTS discontinuity."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    sample_rate: int = Field(gt=0)
    channels: int = Field(ge=1, le=2)
    max_discontinuity_seconds: float = Field(default=0.05, ge=0, allow_inf_nan=False)
