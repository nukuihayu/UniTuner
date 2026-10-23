"""Declare native joint H3 sampling geometry and noise policies without compute imports."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MiniMaxH3SamplingConfig(BaseModel):
    """Select a32-aligned canvas, native17n+5 video or one-frame output and Euler schedule.

    FPS below24 stretches only video generation, with matching audio duration.
    Fine bands retain the native temporal lattice; the model-specific band
    limit is checked against the loaded transformer's actual rotary topology.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    width: int = Field(default=768, gt=0, multiple_of=32)
    height: int = Field(default=1344, gt=0, multiple_of=32)
    frames: int = Field(default=124, gt=0)
    fps: int = Field(default=24, ge=1, le=24)
    temporal_fine_bands: int = Field(default=0, ge=0)
    allow_experimental_duration: bool = False
    steps: int = Field(default=30, gt=0)
    seed: int = 42
    attention_backend: Literal["torch", "kitchen-int8"] = "torch"
    video_shift: float = Field(default=12.0, ge=0.01, le=100)
    audio_shift: float = Field(default=3.0, ge=0.01, le=100)
    visual_condition_clean: float = Field(default=0.999, ge=0, le=1)
    audio_condition_clean: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_timeline(self) -> Self:
        """Reject unsupported temporal grids and unreleased durations before artifact IO."""
        if self.frames == 1:
            if self.fps != 24:
                raise ValueError("MiniMaxH3 single-frame sampling requires24fps without temporal stretch")
        else:
            if self.frames < 5 or (self.frames - 5) % 17:
                raise ValueError("MiniMaxH3 video sampling requires17n+5 frames")
            if not self.allow_experimental_duration and not 5 <= self.frames / self.fps <= 15:
                raise ValueError("MiniMaxH3 duration outside5-15seconds requires allow_experimental_duration")
        if self.temporal_fine_bands and self.fps == 24:
            raise ValueError("MiniMaxH3 temporal_fine_bands requires fps below24")
        return self
