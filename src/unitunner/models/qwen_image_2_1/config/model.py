"""Declare immutable Qwen Image 2.1 single-stream topology."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QwenImage21Config(BaseModel):
    """Describe native patch1 channels, causal conditioning and three rotary axes."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    in_channels: int = Field(default=64, gt=0)
    out_channels: int = Field(default=64, gt=0)
    num_layers: int = Field(default=32, gt=0)
    attention_head_dim: int = Field(default=128, gt=0)
    num_attention_heads: int = Field(default=32, gt=0)
    context_in_dim: int = Field(default=4096, gt=0)
    mlp_ratio: int = Field(default=3, gt=0)
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    eps: float = Field(default=1e-6, gt=0)
    patch_size: Literal[1] = 1
    causal_condition: Literal[True] = True

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Require positive even rotary widths summing to each attention head."""
        if (
            any(width < 2 or width % 2 for width in self.axes_dims_rope)
            or sum(self.axes_dims_rope) != self.attention_head_dim
        ):
            raise ValueError("Qwen Image 2.1 rotary axes must be positive even widths summing to attention_head_dim")
        return self

    @property
    def hidden_size(self) -> int:
        """Return the single-stream hidden width."""
        return self.num_attention_heads * self.attention_head_dim
