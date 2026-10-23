"""Declare published MOSS-Image Base/SFT topology independently of runtime strategies."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MossImageConfig(BaseModel):
    """Pin checkpoint dimensions; Base and SFT use the same transformer architecture."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    variant: Literal["base", "sft", "edit"] = "base"
    dim: int = Field(default=3072, gt=0)
    ffn_dim: int = Field(default=8192, gt=0)
    num_heads: int = Field(default=24, gt=0)
    num_layers: int = Field(default=42, gt=0)
    text_dim: int = Field(default=4096, gt=0)
    freq_dim: int = Field(default=256, gt=0)
    latent_channels: int = Field(default=128, gt=0)
    rope_base: float = Field(default=10000.0, gt=0)
    text_blocks: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Require divisible heads with four rotary axes and an even frequency width."""
        if self.dim % self.num_heads or (self.dim // self.num_heads) % 8 or self.freq_dim % 2:
            raise ValueError("MOSS requires head width divisible by eight and an even time frequency width")
        return self
