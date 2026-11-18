"""Declare native wan datasets policies."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WanSingleFrameDatasetConfig(BaseModel):
    """Bind one/two ordered clean-image sources to explicit native temporal coordinates."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    clean_sources: tuple[str, ...] = Field(min_length=1, max_length=2)
    clean_indices: tuple[int, ...] = Field(min_length=1, max_length=2)
    target_index: int = Field(ge=0, lt=1024)

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        """Preserve source/index binding and reject ambiguous or out-of-table positions."""
        indices = (*self.clean_indices, self.target_index)
        if any(not source for source in self.clean_sources) or len(self.clean_sources) != len(self.clean_indices):
            raise ValueError("Wan single_frame requires matching nonempty clean_sources/clean_indices")
        if any(not 0 <= index < 1024 for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("Wan single_frame requires distinct clean/target indices in[0,1024)")
        return self


class WanDatasetConfig(BaseModel):
    """Select an optional complete stem-matched control image/video directory."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    single_frame: WanSingleFrameDatasetConfig | None = None
