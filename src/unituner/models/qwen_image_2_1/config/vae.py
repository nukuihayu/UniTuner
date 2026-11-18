# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Declare Qwen Image2.1 RGBA VAE geometry and native latent normalization."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QwenImage21VAEConfig(BaseModel):
    """Describe image-only residual VAE topology, including inert temporal weights.

    Input channels count physical RGB/RGBA channels before optional patch folding.
    Encoder/decoder widths may differ. Temporal flags retain native registered
    time convolutions and shortcut scaling, but never authorize video inputs.
    Reduced latent widths require explicit mean/std rather than sliced defaults.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    input_channels: Literal[3, 4] = 4
    base_channels: int = Field(default=96, gt=0)
    decoder_base_channels: int = Field(default=144, gt=0)
    latent_channels: int = Field(default=64, gt=0)
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8, 8)
    layers_per_block: int = Field(default=2, gt=0)
    temporal_downsample: tuple[bool, ...] = (False, True, True, True)
    attention_scales: tuple[float, ...] = ()
    is_residual: bool = True
    patch_size: int = Field(default=1, gt=0)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    latents_std: tuple[float, ...] = (
        3.2001,
        3.2936,
        3.4321,
        3.0091,
        3.1061,
        4.0379,
        4.0705,
        3.791,
        3.0785,
        3.65,
        3.9308,
        3.0904,
        2.8778,
        3.7675,
        3.732,
        5.0756,
        3.2864,
        4.0397,
        3.1317,
        4.0443,
        2.9249,
        3.9454,
        3.0988,
        4.2489,
        3.4896,
        3.8513,
        3.9323,
        3.4719,
        3.7498,
        4.283,
        3.5694,
        4.2467,
        3.9037,
        3.2947,
        5.077,
        3.5075,
        3.27,
        3.4767,
        2.8063,
        5.1125,
        3.5327,
        4.7833,
        3.1286,
        4.1819,
        3.8527,
        3.8312,
        3.5605,
        4.3875,
        3.9624,
        4.0168,
        3.5643,
        4.055,
        5.5614,
        4.2963,
        4.408,
        3.4959,
        3.8747,
        3.7608,
        3.5735,
        3.149,
        3.7662,
        3.6746,
        3.4563,
        3.8161,
    )
    latents_mean: tuple[float, ...] = (
        0.5126,
        0.7721,
        -0.0631,
        1.3506,
        -0.7855,
        -2.1025,
        -0.3458,
        1.3722,
        1.8873,
        -1.7177,
        -0.651,
        0.2732,
        0.7562,
        -0.6163,
        -1.0277,
        3.8363,
        2.021,
        0.0472,
        0.932,
        2.0087,
        2.4954,
        -0.1391,
        -1.4249,
        1.8464,
        -0.5236,
        1.2826,
        3.7046,
        -1.3035,
        2.7286,
        -1.4518,
        -1.9036,
        -1.9955,
        -0.0342,
        -1.0265,
        -0.7636,
        3.0555,
        0.0746,
        -3.0751,
        -0.1076,
        1.7376,
        -1.0914,
        -1.9435,
        -0.2784,
        -1.368,
        0.4809,
        -0.4433,
        0.3764,
        0.5729,
        -2.0595,
        1.096,
        -1.326,
        -2.0211,
        -5.0179,
        0.5275,
        4.0162,
        1.8505,
        0.3026,
        1.9373,
        1.4937,
        0.2632,
        0.5547,
        -1.7121,
        -0.1562,
        0.0304,
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Reject invalid channel folding, stage factors and latent statistics."""
        if (
            not self.channel_multipliers
            or any(value < 1 for value in self.channel_multipliers)
            or len(self.temporal_downsample) != len(self.channel_multipliers) - 1
        ):
            raise ValueError("Qwen Image2.1 vae requires positive stages and one temporal flag per transition")
        if (
            len(self.latents_mean) != self.latent_channels
            or len(self.latents_std) != self.latent_channels
            or any(value <= 0 for value in self.latents_std)
        ):
            raise ValueError("Qwen Image2.1 vae requires matching finite means and positive standard deviations")
        valid_scales = {2.0**-index for index in range(len(self.channel_multipliers))}
        if (
            len(set(self.attention_scales)) != len(self.attention_scales)
            or any(value not in valid_scales for value in self.attention_scales)
            or (self.is_residual and self.attention_scales)
        ):
            raise ValueError("Qwen Image2.1 stage attention requires distinct valid scales and nonresidual topology")
        encoder = [self.base_channels * value for value in (1, *self.channel_multipliers)]
        decoder = [
            self.decoder_base_channels * value
            for value in (self.channel_multipliers[-1], *self.channel_multipliers[::-1])
        ]
        for index in range(len(self.channel_multipliers)):
            transition = index < len(self.channel_multipliers) - 1
            down_factor = 4 * (2 if transition and self.temporal_downsample[index] else 1) if transition else 1
            up_factor = 4 * (2 if self.temporal_downsample[::-1][index] else 1) if transition else 1
            if self.is_residual:
                if encoder[index] * down_factor % encoder[index + 1] or (
                    transition and decoder[index + 1] * up_factor % decoder[index]
                ):
                    raise ValueError(
                        "Qwen Image2.1 residual channel widths must divide native shortcut folding factors"
                    )
            elif transition and decoder[index + 1] % 2:
                raise ValueError("Qwen Image2.1 nonresidual upsample channels must be even")
        return self

    @property
    def spatial_factor(self) -> int:
        """Return pixel-to-latent compression, including optional patch folding."""
        return 2 ** (len(self.channel_multipliers) - 1) * self.patch_size
