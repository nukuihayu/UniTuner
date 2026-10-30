"""Declare common text encoding choices with explicit native capability checks."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unituner.artifacts.formats.precision import FloatingPrecision
from unituner.artifacts.formats.quantization import DEFAULT_GROUP_SIZE, DEFAULT_ROTATION_SIZE, WeightQuantization


class TokenizerSourceConfig(BaseModel):
    """Locate tokenizer/processor assets independently of encoder weights."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    source: str | None = Field(default=None, min_length=1)
    subfolder: str | None = Field(default=None, min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    offline: bool = False

    @model_validator(mode="after")
    def validate_subfolder(self) -> Self:
        """Keep asset selection inside the selected repository or directory."""
        if self.subfolder is not None and (
            self.subfolder.startswith(("/", "\\"))
            or "\\" in self.subfolder
            or any(part in ("", ".", "..") for part in self.subfolder.split("/"))
            or ":" in self.subfolder
        ):
            raise ValueError("Tokenizer subfolder must be a relative repository path without traversal")
        return self


class TextEncodingConfig(BaseModel):
    """Select frozen text computation/storage and supported token budgets.

    Omitted values retain native tokenization and feature geometry. Native
    hidden-state selection belongs to the encoder artifact, not training options.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    tokenizer: TokenizerSourceConfig = Field(default_factory=TokenizerSourceConfig)
    clip_tokenizer: TokenizerSourceConfig | None = None
    processor: TokenizerSourceConfig | None = None
    max_length: int | None = Field(default=None, gt=0)

    precision: FloatingPrecision | None = None
    quantization: WeightQuantization = "none"
    quantization_rotation_size: int = Field(default=DEFAULT_ROTATION_SIZE, ge=4)
    quantization_group_size: int = Field(default=DEFAULT_GROUP_SIZE, ge=4)

    @model_validator(mode="after")
    def validate_quantization(self) -> Self:
        rotation = self.quantization_rotation_size
        if rotation & (rotation - 1) or (rotation.bit_length() - 1) % 2:
            raise ValueError("Text quantization rotation size must be a power of four")
        group = self.quantization_group_size
        if 16 % group and group % 16:
            raise ValueError("Text quantization group size must divide 16 or be a multiple of 16")
        return self

    def resolved_max_length(self, family: str) -> int:
        """Return an explicit token budget for encoders that expose this choice."""
        defaults = {"z-image": 512, "moss-image": 4096, "flux": 512, "krea2": 512}
        if family not in defaults:
            raise ValueError(f"{family} uses a fixed native text budget")
        return self.max_length or defaults[family]

    def validate_for(self, family: str) -> None:
        """Reject unsupported overrides before tokenizers or model weights are loaded."""
        if self.clip_tokenizer is not None and family != "flux":
            raise ValueError("model.text.clip_tokenizer requires FLUX")
        if self.processor is not None and family != "qwen-image":
            raise ValueError("model.text.processor is a separate source only for Qwen Image")
        if self.max_length is not None:
            self.resolved_max_length(family)
            limit = 512 if family == "flux" else 4096 if family == "moss-image" else None
            if limit is not None and self.max_length > limit:
                raise ValueError(f"{family} model.text.max_length must not exceed {limit}")


DEFAULT_TEXT_CONFIG = TextEncodingConfig()
