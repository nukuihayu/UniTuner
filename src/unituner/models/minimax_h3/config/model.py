# Copyright 2025-2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Diffusers PR #14355 (abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc)
# via the pinned Musubi and old MixTuner implementations.
"""Declare strict native MiniMaxH3 denoiser topology and AdaLN artifact variants."""

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MiniMaxH3Config(BaseModel):
    """Describe joint AV attention, independent native precision islands and AdaLN shape."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    in_channels: int = 24
    audio_in_channels: int = 32
    hidden_size: int = Field(default=5376, gt=0)
    num_layers: int = Field(default=50, gt=0)
    token_refiner_num_layers: int = Field(default=2, gt=0)
    num_attention_heads: int = Field(default=56, gt=0)
    attention_head_dim: int = Field(default=128, gt=0)
    ffn_hidden_size: int = Field(default=14336, gt=0)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = Field(default=5120, gt=0)
    timestep_input_dim: int = Field(default=256, gt=0)
    time_embed_hidden_size: int = Field(default=5376, gt=0)
    time_embed_dim: int = Field(default=2688, gt=0)
    rope_inv_freq_len: int = Field(default=16, gt=0)
    norm_eps: float = Field(default=1e-5, gt=0)
    qk_norm_eps: float = Field(default=1e-5, gt=0)
    final_norm_eps: float = Field(default=1e-5, gt=0)
    adaln_curve_grid: int | None = Field(default=None, gt=0)
    adaln_rank: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Preserve native channel/patch/rotary constraints and exclusive AdaLN variants."""
        if self.in_channels != 24 or self.audio_in_channels != 32 or self.patch_size != (1, 2, 2):
            raise ValueError("MiniMaxH3 requires24 video/32 audio channels and1x2x2 patches")
        if self.timestep_input_dim % 2 or self.rope_inv_freq_len * 6 > self.attention_head_dim:
            raise ValueError("MiniMaxH3 requires even timestep width and rotary width<=head width")
        if self.adaln_curve_grid is not None:
            if self.adaln_rank is not None or self.adaln_curve_grid != 1025 or self.time_embed_dim != 8:
                raise ValueError("MiniMaxH3 curve AdaLN requires1025 rows, time_embed_dim8 and no basis rank")
        return self

    @property
    def is_pruned(self) -> bool:
        """Return whether curve-table or basis-projected AdaLN is selected."""
        return self.adaln_curve_grid is not None or self.adaln_rank is not None

    @property
    def adaln_in_features(self) -> int:
        """Return AdaLN input width, including the basis mean's constant channel."""
        return self.adaln_rank + 1 if self.adaln_rank is not None else self.time_embed_dim

    @property
    def attention_inner_dim(self) -> int:
        """Return native attention projection width, independently of residual width."""
        return self.num_attention_heads * self.attention_head_dim

    @property
    def video_patch_dim(self) -> int:
        """Return the native flattened visual patch width."""
        return self.in_channels * math.prod(self.patch_size)

    @property
    def block_adaln_out_features(self) -> int:
        """Return six modulation groups for each of three modalities."""
        return 6 * self.hidden_size * 3

    @property
    def final_adaln_out_features(self) -> int:
        """Return final scale/shift projection width."""
        return 2 * self.hidden_size
