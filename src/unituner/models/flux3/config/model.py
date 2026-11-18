"""Declare native FLUX3 topology and distinct base/flow-map time contracts."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Flux3Config(BaseModel):
    """Pin published image/text branches, shared blocks and four rotary axes.

    The distilled checkpoint is a flow map requiring one destination time per
    sample. Base predicts instantaneous velocity without that extra embedding.
    Reduced valid dimensions are explicit test topologies, never inferred weights.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    variant: Literal["base", "distilled"] = "base"
    hidden_size: int = Field(default=3072, gt=0)
    num_heads: int = Field(default=24, gt=0)
    depth: int = Field(default=5, gt=0)
    depth_single: int = Field(default=28, gt=0)
    mlp_ratio: float = Field(default=3.0, gt=0)
    axes_dim: tuple[int, int, int, int] = (32, 32, 32, 32)
    theta: int = Field(default=10000, gt=0)
    image_channels: int = Field(default=128, gt=0)
    text_channels: int = Field(default=20480, gt=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Reject invalid head, rotary or FFN dimensions before model allocation."""
        if self.hidden_size % self.num_heads or sum(self.axes_dim) != self.hidden_size // self.num_heads:
            raise ValueError("FLUX3 rotary axes must sum to the integral per-head width")
        if any(axis < 2 or axis % 2 for axis in self.axes_dim) or int(self.hidden_size * self.mlp_ratio) < 1:
            raise ValueError("FLUX3 rotary axes must be positive/even and FFN width positive")
        return self
