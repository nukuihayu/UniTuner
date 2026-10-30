"""Declare Qwen Image topology and variant-specific conditioning without Torch."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

type QwenImageVariant = Literal["original", "edit", "edit-2509", "edit-2511", "layered"]


class QwenImageConfig(BaseModel):
    """Describe published original/edit/layered networks with explicit reduced geometry.

    Edit-2511 uses zero-time reference modulation. Layered uses a two-entry
    RGB/RGBA embedding and a negative first rotary axis for its last condition
    grid. These are variant semantics, not independently selectable switches.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    variant: QwenImageVariant = "original"
    patch_size: int = Field(default=2, gt=0)
    in_channels: int = Field(default=64, gt=0)
    out_channels: int = Field(default=16, gt=0)
    num_layers: int = Field(default=60, gt=0)
    attention_head_dim: int = Field(default=128, gt=0)
    num_attention_heads: int = Field(default=24, gt=0)
    joint_attention_dim: int = Field(default=3584, gt=0)
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Require positive even rotary axes and packed patch input/output widths."""
        if any(value < 2 or value % 2 for value in self.axes_dims_rope):
            raise ValueError("Qwen Image rotary axes must be positive even widths")
        if sum(self.axes_dims_rope) != self.attention_head_dim:
            raise ValueError("Qwen Image rotary axes must sum to attention_head_dim")
        if self.in_channels != self.patch_size**2 * self.out_channels:
            raise ValueError("Qwen Image input width must equal patch_size squared times out_channels")
        return self

    @property
    def hidden_size(self) -> int:
        """Return joint image/text stream width."""
        return self.num_attention_heads * self.attention_head_dim

    @property
    def zero_reference_time(self) -> bool:
        """Select the published Edit-2511 reference-time modulation."""
        return self.variant == "edit-2511"
