"""Declare native H3 generation inputs and local artifacts without compute or media IO."""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from unituner.config.generation import GenerationAdapterConfig
from unituner.config.model import ModelConfig
from unituner.config.runtime import RuntimeConfig
from unituner.models.minimax_h3.config.inputs import MiniMaxH3SampleConfig, validate_minimax_h3_inputs


class MiniMaxH3GenerationConfig(MiniMaxH3SampleConfig):
    """Configure local FP32 H3 generation with PNG single-frame or H264/AAC MP4/MKV output."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig
    adapter: GenerationAdapterConfig | None = None
    output: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        """Require complete local artifact declarations and validate task/output before IO."""
        model = self.model
        if model.family != "minimax-h3" or model.vae is None or model.text_encoder is None or model.minimax_h3 is None:
            raise ValueError("MiniMaxH3 generation requires model vae/text_encoder/minimax_h3 artifacts")
        validate_minimax_h3_inputs(self.inputs, self.sampling, task=model.variant)
        legal = (".png",) if self.sampling.frames == 1 else (".mp4", ".mkv")
        if Path(self.output).suffix.lower() not in legal:
            raise ValueError("MiniMaxH3 output requires PNG for a single-frame or MP4/MKV for video")
        return self
