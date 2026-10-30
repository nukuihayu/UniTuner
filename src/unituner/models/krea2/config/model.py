"""Declare Krea2 topology and native grouped-query rotary constraints."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Krea2Config(BaseModel):
    """Describe RAW/Turbo shared geometry with fixed two-plus-two text fusion blocks.

    text_layers counts selected encoder hidden states per token, not transformer
    depth. Native parameter keys remain independent of these configuration names.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    variant: Literal["raw", "turbo"] = "raw"
    hidden_size: int = Field(default=6144, gt=0)
    timestep_dim: int = Field(default=256, ge=2)
    text_dim: int = Field(default=2560, gt=0)
    num_heads: int = Field(default=48, gt=0)
    num_kv_heads: int = Field(default=12, gt=0)
    mlp_multiplier: int = Field(default=4, gt=0)
    num_layers: int = Field(default=28, gt=0)
    patch_size: int = Field(default=2, gt=0)
    in_channels: int = Field(default=16, gt=0)
    bias: bool = False
    theta: float = Field(default=1000.0, gt=0)
    text_layers: int = Field(default=12, gt=0)
    text_heads: int = Field(default=20, gt=0)
    text_kv_heads: int = Field(default=20, gt=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Reject invalid head/group/time and positive even three-axis rotary geometry."""
        if (
            self.hidden_size % self.num_heads
            or self.text_dim % self.text_heads
            or self.num_heads % self.num_kv_heads
            or self.text_heads % self.text_kv_heads
            or self.timestep_dim % 2
        ):
            raise ValueError("Krea2 requires integral head widths, divisible query/KV groups and even timestep_dim")
        if any(axis < 1 or axis % 2 for axis in self.rotary_axes):
            raise ValueError("Krea2 requires positive even native rotary axes")
        return self

    @property
    def rotary_axes(self) -> tuple[int, int, int]:
        """Return native temporal/height/width rotary widths for one visual head."""
        width = self.hidden_size // self.num_heads
        return (width - 12 * (width // 16), 6 * (width // 16), 6 * (width // 16))
