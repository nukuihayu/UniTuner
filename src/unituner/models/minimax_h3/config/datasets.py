"""Declare native minimax h3 datasets policies."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unituner.config.audio import AudioDecodeConfig


class MiniMaxH3DatasetConfig(BaseModel):
    """Pin posterior sampling and32kHz stereo decode/repair for JSONL AV sources."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    cache_seed: int = 0
    allow_experimental_duration: bool = False
    audio_decode: AudioDecodeConfig = Field(default_factory=lambda: AudioDecodeConfig(sample_rate=32000, channels=2))
    audio_pad_tolerance: int = Field(default=800, ge=0)
    audio_max_repair_samples: int = Field(default=6400, ge=0)

    @model_validator(mode="after")
    def validate_audio_clock(self) -> Self:
        """Keep the released native audio clock for target and reference encoders."""
        if self.audio_decode.sample_rate != 32000 or self.audio_decode.channels != 2:
            raise ValueError("MiniMaxH3 cache requires32kHz stereo audio")
        return self
