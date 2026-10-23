# Copyright (c) 2025 Z-Image Team. Apache-2.0.
# Adapted from https://github.com/Tongyi-MAI/Z-Image via Musubi Tuner.
# Original vae derived from https://github.com/black-forest-labs/flux.
# Modified to provide explicit posterior noise and typed native topology.
"""Declare native z image VAE configuration."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ZImageVAEConfig(BaseModel):
    """Describe VAE geometry and model-latent normalization; defaults are native."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    in_channels: int = Field(default=3, gt=0)
    out_channels: int = Field(default=3, gt=0)
    block_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = Field(default=2, gt=0)
    latent_channels: int = Field(default=16, gt=0)
    norm_groups: int = Field(default=32, gt=0)
    scaling_factor: float = Field(default=0.3611, gt=0)
    shift_factor: float = 0.1159
    quant_conv: bool = False
    post_quant_conv: bool = False

    @model_validator(mode="after")
    def validate_channels(self) -> Self:
        """Require positive VAE widths divisible by GroupNorm group count."""
        if not self.block_channels or any(width <= 0 or width % self.norm_groups for width in self.block_channels):
            raise ValueError("VAE block widths must be positive multiples of norm_groups")
        return self
