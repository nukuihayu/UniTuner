"""Declare feature objectives independently of routing and execution policies."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LayerSyncConfig(BaseModel):
    """Align an earlier visual layer with detached later-layer target features."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    weight: float = Field(default=1.0, gt=0)
    student_layer: int = Field(default=0, ge=0)
    teacher_layer: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_layers(self) -> Self:
        """Require a strictly later teacher in the native visual block order."""
        if self.teacher_layer <= self.student_layer:
            raise ValueError("LayerSync teacher_layer must be after student_layer")
        return self


class RepaConfig(BaseModel):
    """Align a selected native target layer with identity-pinned cached encoder features."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    weight: float = Field(default=1.0, gt=0)
    layer: int = Field(default=0, ge=0)
    encoder_dim: int = Field(gt=0)
    encoder_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_directory: str = Field(min_length=1)
    encoder_weights: str | None = None
    encoder_device: str = "cpu"
