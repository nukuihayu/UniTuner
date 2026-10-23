"""Validate native H3 video convolution, temporal overlap and decoder topology."""

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MiniMaxH3VideoDecoderConfig(BaseModel):
    """Declare native rotary transformer decoder topology, independent of tiling."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)

    num_layers: int = Field(default=36, gt=0)
    num_heads: int = Field(default=32, gt=0)
    head_dim: int = Field(default=64, gt=0)
    rope_theta: float = Field(default=100.0, gt=0)
    rope_dim_ratio: float = Field(default=0.75, gt=0, le=1)
    bias: bool = True
    eps: float = Field(default=1e-5, gt=0)
    num_register_tokens: int = Field(default=4, ge=0)

    @model_validator(mode="after")
    def validate_rotary_dimension(self) -> Self:
        """Require a nonempty even three-axis rotary prefix within each head."""
        rotary_dim = int(self.head_dim * self.rope_dim_ratio)
        if rotary_dim < 6 or rotary_dim % 6:
            raise ValueError("H3 video decoder rotary dimension must be positive and divisible by 6")
        return self


class MiniMaxH3VideoVAEConfig(BaseModel):
    """Describe native 24-channel video moments and exact clip/overlap geometry.

    Released geometry encodes 17 raw frames into 5 tokens before dropping the
    final 3 tokens over a whole clip sequence. Spatial compression is 16.
    Decoder topology is retained even when its weights are explicitly omitted.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)

    channels: int = Field(default=128, gt=0)
    channel_multipliers: tuple[int, ...] = (1, 2, 2, 4, 4, 8)
    residual_blocks: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    spatial_strides: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    temporal_strides: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    clip_length: int = Field(default=17, gt=0)
    token_drop: int = Field(default=3, ge=0)
    tile_size: int = Field(default=256, gt=0)
    tile_overlap: int = Field(default=64, gt=0)
    tiling: bool = True
    decoder: MiniMaxH3VideoDecoderConfig = Field(default_factory=MiniMaxH3VideoDecoderConfig)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Reject invalid level inventories, group widths and nonterminating tile splits."""
        levels = len(self.channel_multipliers)
        arrays = (self.channel_multipliers, self.residual_blocks, self.spatial_strides, self.temporal_strides)
        if not levels or any(len(values) != levels for values in arrays):
            raise ValueError("H3 video encoder level inventories must be nonempty and equal length")
        if any(value <= 0 for values in arrays for value in values):
            raise ValueError("H3 video encoder level values must be positive integers")
        if self.channels % 32 or any(stride not in (1, 2) for stride in self.spatial_strides):
            raise ValueError("H3 video encoder requires 32-group widths and spatial strides 1 or 2")
        if any(stride not in (1, 2) for stride in self.temporal_strides):
            raise ValueError("H3 video temporal strides must be 1 or 2")
        if any(
            time > 1 and space == 1 for time, space in zip(self.temporal_strides, self.spatial_strides, strict=True)
        ):
            raise ValueError("H3 video temporal downsampling requires native simultaneous spatial downsampling")
        spatial_ratio = math.prod(self.spatial_strides)
        if self.tile_size % spatial_ratio or self.tile_overlap % spatial_ratio or self.tile_overlap >= self.tile_size:
            raise ValueError("H3 video tiles require aligned size/overlap with overlap smaller than size")
        if self.token_drop >= math.ceil(self.clip_length / math.prod(self.temporal_strides)):
            raise ValueError("H3 video token_drop must be smaller than one encoded clip")
        return self
