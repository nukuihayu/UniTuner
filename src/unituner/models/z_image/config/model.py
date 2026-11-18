"""Declare exact Z-Image topology independently of runtime policies."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ZImageConfig(BaseModel):
    """Own native dimensions; defaults describe the released base/Turbo topology.

    RoPE uses three positive even axis widths summing to the head width.
    Patch presets pair spatial and temporal sizes; no Cartesian product is implied.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    variant: Literal["base", "turbo"] = "base"
    in_channels: int = Field(default=16, gt=0)
    dim: int = Field(default=3840, gt=0)
    num_layers: int = Field(default=30, gt=0)
    num_refiner_layers: int = Field(default=2, gt=0)
    num_heads: int = Field(default=30, gt=0)
    num_kv_heads: int = Field(default=30, gt=0)
    norm_eps: float = Field(default=1e-5, gt=0)
    qk_norm: bool = True
    caption_dim: int = Field(default=2560, gt=0)
    timestep_hidden_dim: int = Field(default=1024, gt=0)
    timestep_frequency_dim: int = Field(default=256, ge=2)
    time_scale: float = Field(default=1000.0, gt=0)
    rope_theta: float = Field(default=256.0, gt=0)
    axes_dims: tuple[int, int, int] = (32, 48, 48)
    axes_lengths: tuple[int, int, int] = (1536, 512, 512)
    patch_sizes: tuple[int, ...] = (2,)
    frame_patch_sizes: tuple[int, ...] = (1,)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Reject incompatible heads, RoPE axes and duplicate patch presets."""
        if self.dim % self.num_heads or self.num_heads % self.num_kv_heads:
            raise ValueError("Head widths must divide dim and query heads must divide into KV groups")
        if any(width <= 0 or width % 2 for width in self.axes_dims):
            raise ValueError("RoPE axis widths must be positive and even")
        if sum(self.axes_dims) != self.dim // self.num_heads or min(self.axes_lengths) <= 0:
            raise ValueError("RoPE axes must sum to head width and have positive lengths")
        pairs = tuple(zip(self.patch_sizes, self.frame_patch_sizes, strict=True))
        if not pairs or len(pairs) != len(set(pairs)) or any(min(pair) <= 0 for pair in pairs):
            raise ValueError("Patch presets must be nonempty, positive and unique")
        return self
