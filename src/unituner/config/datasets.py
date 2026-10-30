"""Declare dataset sources independently of cache production."""

import json
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import InitErrorDetails

from unituner.config.dataset_inputs import DatasetInputConfig
from unituner.config.diagnostics import ConfigRuleError
from unituner.data.degradation.config import DegradationConfig
from unituner.models.minimax_h3.config.datasets import MiniMaxH3DatasetConfig
from unituner.models.wan.config.datasets import WanDatasetConfig

type Resolution = tuple[int, int]
type Resolutions = Resolution | tuple[Resolution, ...]


def resolution_scales(value: Resolutions) -> tuple[Resolution, ...]:
    """Return ordered, unique positive scale pairs without accepting an empty list."""
    scales = (
        (cast(Resolution, value),)
        if len(value) == 2 and isinstance(value[0], int)
        else cast(tuple[Resolution, ...], value)
    )
    if not scales or any(any(dimension <= 0 for dimension in scale) for scale in scales):
        raise ConfigRuleError(
            "resolution",
            "Dataset resolutions require nonempty positive width/height pairs",
            "Use [1024,1024] or [[1024,1024],[768,768]].",
        )
    if len(set(scales)) != len(scales):
        raise ConfigRuleError("resolution", "Dataset resolutions must be unique", "Remove duplicate resolution pairs.")
    return scales


class DatasetSamplingPolicy(BaseModel):
    """Share caption, bucket and batching policies across input and resolved groups."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    batch_size: int = Field(default=1, gt=0)
    repeats: int = Field(default=1, gt=0)
    enable_bucket: bool = True
    bucket_no_upscale: bool = False
    prompt: str | None = Field(
        default=None, description="Override every sample caption, including with an empty string"
    )
    caption_extension: str | None = Field(default=None, pattern=r"^\.[A-Za-z0-9]+$")


class DatasetSamplingConfig(DatasetSamplingPolicy):
    """Declare sampling defaults with one resolution or an ordered scale list."""

    resolution: Resolutions = (1024, 1024)


class VideoDatasetConfig(BaseModel):
    """Choose output-grid windows and explicit native fractional or timestamp resampling."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    target_frames: tuple[int, ...] = (81,)
    extraction: Literal["head", "chunk", "slide", "uniform", "full"] = "head"
    stride: int = Field(default=1, gt=0)
    samples: int = Field(default=1, gt=0)
    max_frames: int = Field(default=129, gt=0)
    source_fps: float | None = Field(default=None, gt=0)
    target_fps: float = Field(default=16.0, gt=0)
    fps_resampling: Literal["fractional", "timestamps"] = "fractional"

    @model_validator(mode="after")
    def validate_frames(self) -> Self:
        """Reject invalid declared counts and contradictory timestamp rate sources."""
        if not self.target_frames or any(value < 1 for value in self.target_frames):
            raise ConfigRuleError(
                "target_frames",
                "video.target_frames requires nonempty positive frame counts",
                "Supply at least one positive frame count, e.g. [81].",
            )
        if self.fps_resampling == "timestamps" and self.source_fps is not None:
            raise ConfigRuleError(
                "source_fps",
                "Timestamp resampling uses decoder timestamps, not video.source_fps",
                "Remove source_fps when fps_resampling='timestamps'.",
            )
        return self


class CacheDatasetConfig(DatasetInputConfig, DatasetSamplingPolicy):
    """Describe source and cache paths relative to the caller's working directory."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    resolution: Resolution = (1024, 1024)
    kind: Literal["image", "video", "restoration"] = "image"
    video: VideoDatasetConfig | None = None
    wan: WanDatasetConfig | None = None
    minimax_h3: MiniMaxH3DatasetConfig | None = None
    source: str = Field(min_length=1)
    cache_directory: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cache_geometry(self) -> Self:
        """Check only geometry and media layout needed by encoders."""
        if any(dimension <= 0 for dimension in self.resolution):
            raise ConfigRuleError(
                "resolution",
                "dataset.resolution dimensions must be positive",
                "Set width and height to positive integers, e.g. [1024,1024].",
            )
        if self.video is not None and self.kind != "video":
            raise ConfigRuleError(
                "video", "dataset.video requires video data", "Set kind='video' or remove this video section."
            )
        return self


class DatasetConfig(CacheDatasetConfig):
    """Add training-only aligned/synthetic restoration inputs."""

    low_quality_source: str | None = None
    degradation: DegradationConfig | None = None
    horizontal_flip_probability: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_restoration_sources(self) -> Self:
        """Reject incomplete or ambiguous restoration training inputs."""
        synthetic = self.degradation is not None and self.degradation.enabled
        if synthetic and self.degradation.output_mode != "aligned":
            raise ValueError("Restoration datasets require degradation.output_mode='aligned'; native is for previews")
        if self.kind != "restoration" and (self.degradation is not None or self.low_quality_source is not None):
            raise ConfigRuleError(
                "kind",
                "Low-quality sources and degradation are only valid for image restoration",
                "Set kind='restoration' or remove low_quality_source and degradation.",
            )
        if synthetic and self.low_quality_source is not None:
            raise ConfigRuleError(
                "low_quality_source",
                "Synthetic degradation and low_quality_source are mutually exclusive",
                "Remove low_quality_source for synthetic degradation, or disable degradation.enabled.",
            )
        if synthetic and min(self.resolution) < 32:
            raise ConfigRuleError(
                "resolution",
                "Synthetic degradation requires resolution dimensions at least 32",
                "Set both dimensions to at least 32 for synthetic degradation.",
            )
        if self.kind == "restoration" and not synthetic and not self.low_quality_source:
            raise ConfigRuleError(
                "low_quality_source",
                "low_quality_source is required for image restoration data",
                "Supply the aligned LQ directory, or enable synthetic degradation and omit low_quality_source.",
            )
        if self.kind != "restoration" and self.horizontal_flip_probability != 0.5:
            raise ConfigRuleError(
                "horizontal_flip_probability",
                "Explicit horizontal_flip_probability is supported only for image restoration",
                "Use kind='restoration' or leave horizontal_flip_probability at its default of 0.5.",
            )
        return self


class DatasetGroupsConfig[D: CacheDatasetConfig](BaseModel):
    """Resolve named independent datasets over shared sampling defaults before IO."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    defaults: DatasetSamplingConfig = Field(default_factory=DatasetSamplingConfig)
    groups: dict[str, D] = Field(min_length=1)

    @field_validator("groups", mode="before")
    @classmethod
    def inherit_defaults(cls, groups: Any, info: ValidationInfo) -> Any:
        """Resolve defaults using JSON validation so strict tuple fields retain array input semantics."""
        defaults = info.data.get("defaults")
        if not isinstance(groups, dict) or not isinstance(defaults, DatasetSamplingConfig):
            return groups
        merged = {
            name: {**defaults.model_dump(mode="json"), **group} if isinstance(group, dict) else group
            for name, group in groups.items()
        }
        if all(isinstance(group, CacheDatasetConfig) for group in merged.values()):
            return merged
        expanded = {}
        for name, group in merged.items():
            if not isinstance(group, dict):
                expanded[name] = group
                continue
            try:
                resolution = TypeAdapter(Resolutions).validate_json(json.dumps(group["resolution"]), strict=True)
            except ValidationError as error:
                raise ValidationError.from_exception_data(
                    cls.__name__,
                    [
                        cast(InitErrorDetails, {**item, "loc": (name, "resolution", *item["loc"])})
                        for item in error.errors(include_url=False)
                    ],
                ) from None
            nested = not resolution or isinstance(resolution[0], tuple)
            if not nested:
                expanded[name] = group
                continue
            try:
                scales = resolution_scales(resolution)
            except ConfigRuleError as error:
                raise ConfigRuleError(f"{name}.resolution", str(error), error.hint) from None
            for scale in scales:
                key = f"{name}__{scale[0]}x{scale[1]}" if nested else name
                if key in expanded or (key != name and key in groups):
                    raise ConfigRuleError(
                        "groups",
                        "Expanded resolution group names collide",
                        "Rename groups that overlap generated NAME__WIDTHxHEIGHT identifiers.",
                    )
                resolved = {**group, "resolution": list(scale)}
                if nested and isinstance(group.get("cache_directory"), str):
                    resolved["cache_directory"] = str(Path(group["cache_directory"]) / f"{scale[0]}x{scale[1]}")
                expanded[key] = resolved
        annotation = cls.model_fields["groups"].annotation
        assert annotation is not None
        return TypeAdapter(annotation).validate_json(json.dumps(expanded), strict=True)

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        """Keep names addressable from CLI and prevent multiple publishers sharing a cache."""
        if any(not name.isidentifier() or not name.isascii() for name in self.groups):
            raise ConfigRuleError(
                "groups",
                "Data group names must be ASCII identifiers (letters, digits and underscores)",
                "Use a name starting with an ASCII letter or underscore, followed by letters, digits or underscores.",
            )
        caches = [Path(group.cache_directory).resolve() for group in self.groups.values()]
        if len(set(caches)) != len(caches):
            raise ConfigRuleError(
                "groups",
                "Each data group requires a distinct resolved cache_directory",
                "Assign a separate cache_directory to every group; relative aliases resolving to the same directory are not independent caches.",
            )
        return self


class DataConfig(DatasetGroupsConfig[DatasetConfig]):
    """Resolve complete training datasets and restoration source policies."""


class CacheDataConfig(DatasetGroupsConfig[CacheDatasetConfig]):
    """Resolve cache groups without training-only restoration requirements."""
