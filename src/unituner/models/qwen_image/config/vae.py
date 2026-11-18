"""Declare Qwen Image causal RGB/RGBA VAE geometry and channel normalization."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QwenImageVAEConfig(BaseModel):
    """Describe published spatial8/temporal4 topology and raw latent statistics.

    RGBA layered weights use input_channels4 with the same latent topology.
    Reduced latent widths require explicit matching mean/std, never a slice of
    production normalization constants. Independent image layers are not video.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    input_channels: Literal[3, 4] = 3
    base_channels: int = Field(default=96, gt=0)
    latent_channels: int = Field(default=16, gt=0)
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    layers_per_block: int = Field(default=2, gt=0)
    temporal_downsample: tuple[bool, ...] = (False, True, True)
    attention_scales: tuple[float, ...] = ()
    dropout: float = Field(default=0.0, ge=0, lt=1)
    latents_mean: tuple[float, ...] = (
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    )
    latents_std: tuple[float, ...] = (
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.916,
    )

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        """Validate stream widths, compression flags and finite channel statistics."""
        if (
            not self.channel_multipliers
            or self.base_channels % 2
            or any(value < 1 for value in self.channel_multipliers)
            or len(self.temporal_downsample) != len(self.channel_multipliers) - 1
            or sum(self.temporal_downsample) != 2
        ):
            raise ValueError("Qwen Image vae requires even base width, positive stages and temporal compression4")
        valid_scales = {2.0**-index for index in range(len(self.channel_multipliers))}
        if len(set(self.attention_scales)) != len(self.attention_scales) or any(
            value not in valid_scales for value in self.attention_scales
        ):
            raise ValueError("Qwen Image vae attention scales must be distinct encoder stage scales")
        if (
            len(self.latents_mean) != self.latent_channels
            or len(self.latents_std) != self.latent_channels
            or any(value <= 0 for value in self.latents_std)
        ):
            raise ValueError("Qwen Image vae requires one mean and positive std per latent channel")
        return self

    @property
    def spatial_factor(self) -> int:
        """Return native spatial compression from the number of encoder transitions."""
        return 2 ** (len(self.channel_multipliers) - 1)
