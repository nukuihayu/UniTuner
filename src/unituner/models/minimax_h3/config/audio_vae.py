"""Validate native H3 stereo DAC and BigVGAN VAE dimensions."""

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MiniMaxH3AudioVAEConfig(BaseModel):
    """Describe native 32kHz stereo, 800-sample-hop, 32-channel mean encoding.

    Decoder upsampling retains the released seven-stage 800x preset. Reduced
    feature widths preserve eight attention heads and native pooling direction.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)

    encoder_channels: int = Field(default=64, gt=0)
    encoder_strides: tuple[int, ...] = (2, 4, 4, 5, 5)
    hidden_channels: int = Field(default=2048, ge=256)
    decoder_channels: int = Field(default=1024, ge=128)

    @model_validator(mode="after")
    def validate_native_geometry(self) -> Self:
        """Reject inconsistent native hop, head widths and decoder channel halving."""
        if (
            not self.encoder_strides
            or any(value < 2 for value in self.encoder_strides)
            or math.prod(self.encoder_strides) != 800
        ):
            raise ValueError("H3 audio encoder strides must be integers >=2 with product 800")
        if self.hidden_channels % 8:
            raise ValueError("H3 audio hidden channels must divide evenly into eight attention heads")
        if self.decoder_channels % 128:
            raise ValueError("H3 audio decoder channels must divide evenly over seven halving stages")
        return self
