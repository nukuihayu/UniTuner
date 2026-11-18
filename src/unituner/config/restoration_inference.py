"""Declare one-step image restoration inputs without importing compute backends."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unituner.config.generation import GenerationAdapterConfig
from unituner.config.model import ModelConfig
from unituner.config.runtime import RuntimeConfig
from unituner.models.catalog import ModelFamily


class RestorationInferenceConfig(BaseModel):
    """Select local artifacts and a low-quality image, preserving its oriented size.

    Sigma must match training. Reflection padding supplies the native spatial
    granularity; output is cropped back without resampling the source image.
    Encoder weights are optional only for models trained without encoder adapters.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: ModelConfig
    input: str = Field(min_length=1)
    output: str = Field(min_length=1)
    prompt: str = ""
    attention_backend: Literal["torch", "kitchen-int8"] = "torch"
    sigma: float | None = Field(default=None, gt=0, lt=1)
    adapter: GenerationAdapterConfig | None = None
    encoder_weights: str | None = Field(default=None, min_length=1)
    reference_images: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        if self.reference_images and self.model.family != ModelFamily.MINIMAX_H3_IMAGE:
            raise ValueError("Restoration reference_images currently require MiniMax-H3-Image")
        if len(self.reference_images) > 16 or any(not path for path in self.reference_images):
            raise ValueError("H3-Image restoration supports up to 16 nonempty references")
        return self
