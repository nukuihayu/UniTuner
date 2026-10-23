"""Declare native Krea2 text feature geometry carried by encoder artifacts."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Krea2TextFeatures(BaseModel):
    """Pin token budget and ordered decoder-input states used by trained weights.

    Published Krea2 uses twelve states at indices2,5,...,35. Nonstandard native
    artifacts declare their geometry in text_config.unitunner_features inside
    the encoder's config.json. This is an artifact contract, not a train flag.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    max_length: int = Field(default=512, gt=0)
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

    @model_validator(mode="after")
    def validate_states(self) -> Self:
        """Preserve feature order while rejecting empty or negative state indices."""
        if not self.select_layers or min(self.select_layers) < 0:
            raise ValueError("Krea2 text features require nonempty nonnegative hidden-state indices")
        return self
