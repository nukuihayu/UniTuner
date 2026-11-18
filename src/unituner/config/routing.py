"""Declare TREAD target-token selection independently of feature objectives."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TreadConfig(BaseModel):
    """Select a fixed subset of target tokens across a half-open native visual window."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    drop_fraction: float = Field(default=0.5, gt=0, lt=1)
    start_layer: int = Field(default=0, ge=0)
    end_layer: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        """Require a nonempty half-open layer window."""
        if self.start_layer >= self.end_layer:
            raise ValueError("TREAD requires start_layer < end_layer")
        return self
