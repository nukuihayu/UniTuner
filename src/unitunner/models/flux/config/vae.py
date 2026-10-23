# Adapted from Black Forest Labs FLUX, Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""Declare native flux VAE configuration."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FluxVAEConfig(BaseModel):
    """Pin native convolution widths and scalar latent normalization."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    in_channels: int = Field(default=3, gt=0)
    out_channels: int = Field(default=3, gt=0)
    base_channels: int = Field(default=128, gt=0)
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    layers_per_block: int = Field(default=2, gt=0)
    latent_channels: int = Field(default=16, gt=0)
    norm_groups: int = Field(default=32, gt=0)
    scaling_factor: float = Field(default=0.3611, gt=0)
    shift_factor: float = 0.1159

    @model_validator(mode="after")
    def validate_channels(self) -> Self:
        """Require positive GroupNorm-compatible widths, before tensor allocation."""
        if (
            not self.channel_multipliers
            or self.base_channels % self.norm_groups
            or any(
                multiplier < 1 or self.base_channels * multiplier % self.norm_groups
                for multiplier in self.channel_multipliers
            )
        ):
            raise ValueError("FLUX vae widths must be positive multiples of norm_groups")
        return self

    @property
    def spatial_factor(self) -> int:
        """Return the pixel-to-latent factor, excluding transformer token packing."""
        return 2 ** (len(self.channel_multipliers) - 1)
