"""Declare immutable Wan 2.1/2.2 denoiser geometry."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WanConfig(BaseModel):
    """Describe a single T2V/I2V expert; expert scheduling belongs to sampling.

    The input channel count includes any concatenated image/control conditions.
    First/last-frame conditioning has its own variant and learned CLIP position
    embedding. It is not represented by the ordinary I2V variant.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    version: Literal["2.1", "2.2"] = "2.1"
    variant: Literal["t2v", "i2v", "flf2v"] = "t2v"
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_length: int = Field(default=512, gt=0)
    in_channels: int = Field(default=16, gt=0)
    out_channels: int = Field(default=16, gt=0)
    hidden_size: int = Field(default=5120, gt=0)
    ffn_dim: int = Field(default=13824, gt=0)
    timestep_dim: int = Field(default=256, ge=2)
    text_dim: int = Field(default=4096, gt=0)
    num_heads: int = Field(default=40, gt=0)
    num_layers: int = Field(default=40, gt=0)
    qk_norm: bool = True
    cross_attention_norm: bool = True
    norm_eps: float = Field(default=1e-6, gt=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Require integral heads and positive even temporal/spatial rotary axes."""
        width = self.hidden_size // self.num_heads
        if self.hidden_size % self.num_heads or width % 2 or width < 6:
            raise ValueError("Wan requires integral even head widths of at least 6")
        if self.timestep_dim % 2 or min(self.patch_size) < 1:
            raise ValueError("Wan requires even timestep_dim and positive patch dimensions")
        if self.variant == "flf2v" and (self.version != "2.1" or self.text_length != 512):
            raise ValueError("Wan FLF2V requires version2.1 and512 text tokens")
        return self

    @property
    def image_context(self) -> bool:
        """Return whether this topology owns a CLIP image branch."""
        return self.version == "2.1" and self.variant in ("i2v", "flf2v")

    @property
    def image_token_count(self) -> int:
        """Return the native image/text split, including both FLF2V images."""
        return (514 if self.variant == "flf2v" else 257) if self.image_context else 0
