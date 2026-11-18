"""Declare strict H3 target, control and reference media inputs without decoding imports."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _ReferenceInput(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    path: str = Field(min_length=1)


class MiniMaxH3ImageReferenceInput(_ReferenceInput):
    """Declare one still image; audio fields are forbidden."""

    kind: Literal["image"]


class MiniMaxH3AudioReferenceInput(_ReferenceInput):
    """Declare standalone audio using its own timestamp origin."""

    kind: Literal["audio"]


class MiniMaxH3VideoReferenceInput(_ReferenceInput):
    """Declare video with optional embedded/independent audio or audio_enabled=false.

    Omitted audio_path selects embedded audio. Explicit null also disables audio
    in JSON; audio_enabled=false expresses that choice in TOML without null.
    """

    kind: Literal["video"]
    audio_path: str | None = Field(default=None, min_length=1)
    audio_enabled: bool = True

    @model_validator(mode="after")
    def validate_audio_selection(self) -> Self:
        """Reject an explicit audio file combined with disabled audio before media IO."""
        if not self.audio_enabled and self.audio_path is not None:
            raise ValueError("MiniMaxH3 video audio_enabled=false cannot select audio_path")
        return self


type MiniMaxH3ReferenceInput = Annotated[
    MiniMaxH3ImageReferenceInput | MiniMaxH3AudioReferenceInput | MiniMaxH3VideoReferenceInput,
    Field(discriminator="kind"),
]


class MiniMaxH3ControlInput(_ReferenceInput):
    """Bind one ordered single-frame FL image to its native24Hz coordinate."""

    index: int = Field(ge=0)


class MiniMaxH3RecordInput(BaseModel):
    """Describe one image/video target and its ordered references in native JSONL.

    Paths resolve against the manifest directory. Empty captions are valid;
    teacher captions must contain non-whitespace text when supplied. Task and
    one-frame geometry are dataset choices, validated during resolution.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    target_path: str = Field(min_length=1)
    caption: str
    references: tuple[MiniMaxH3ReferenceInput, ...] = Field(default=(), max_length=12)
    teacher_caption: str | None = None
    target_audio_path: str | None = Field(default=None, min_length=1)
    target_index: int | None = Field(default=None, ge=0)
    controls: tuple[MiniMaxH3ControlInput, ...] = ()

    @field_validator("teacher_caption")
    @classmethod
    def _teacher_caption(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("teacher_caption must contain non-whitespace text")
        return value
