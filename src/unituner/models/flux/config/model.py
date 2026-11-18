"""Declare FLUX.1 topology without conflating Dev, Kontext and Fill conditioning."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FluxConfig(BaseModel):
    """Pin native packed-token dimensions and external checkpoint key geometry.

    Fill concatenates target, masked-image and mask features along channels;
    Kontext appends reference tokens. Their conditioning stays distinct even
    though Dev and Kontext checkpoints share parameter shapes.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    variant: Literal["dev", "kontext", "fill"] = "dev"
    in_channels: int = Field(default=64, gt=0)
    out_channels: int = Field(default=64, gt=0)
    vec_in_dim: int = Field(default=768, gt=0)
    context_in_dim: int = Field(default=4096, gt=0)
    hidden_size: int = Field(default=3072, gt=0)
    mlp_ratio: float = Field(default=4.0, gt=0)
    num_heads: int = Field(default=24, gt=0)
    depth: int = Field(default=19, gt=0)
    depth_single_blocks: int = Field(default=38, gt=0)
    axes_dim: tuple[int, int, int] = (16, 56, 56)
    theta: int = Field(default=10000, gt=0)
    qkv_bias: bool = True
    guidance_embed: bool = True

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Reject invalid rotary axes and variant channel relationships early."""
        if self.hidden_size % self.num_heads or sum(self.axes_dim) != self.hidden_size // self.num_heads:
            raise ValueError("FLUX rotary axes must sum to the integral head width")
        if any(axis <= 0 or axis % 2 for axis in self.axes_dim):
            raise ValueError("FLUX rotary axes must be positive and even")
        if int(self.hidden_size * self.mlp_ratio) < 1:
            raise ValueError("FLUX MLP width must be positive")
        expected = self.out_channels * (6 if self.variant == "fill" else 1)
        if self.in_channels != expected:
            raise ValueError("FLUX input width must equal output width (Dev/Kontext) or six times output width (Fill)")
        return self
