"""Strict immutable OMGSR-F-1024 degradation parameters."""

from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def _range(value: tuple[float, float]) -> tuple[float, float]:
    if value[0] < 0 or value[0] > value[1]:
        raise ValueError("Expected a nonnegative ordered range")
    return value


def _scale_range(value: tuple[float, float]) -> tuple[float, float]:
    if not 1 <= value[0] <= value[1] <= 16:
        raise ValueError("Scale range must be ordered within [1,16]")
    return value


def _probabilities(value: tuple[float, ...]) -> tuple[float, ...]:
    if min(value) < 0 or abs(sum(value) - 1) > 1e-12:
        raise ValueError("Probabilities must be nonnegative and sum to one")
    return value


type Range = Annotated[tuple[float, float], AfterValidator(_range)]
type Scale = Annotated[float, Field(ge=1, le=16)] | Annotated[tuple[float, float], AfterValidator(_scale_range)]
type Probabilities = Annotated[tuple[float, float, float], AfterValidator(_probabilities)]
type Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class DistributionConfig(BaseModel, frozen=True):
    """Reject NaN, infinity and parameters outside the OMGSR schema."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)


class BlurConfig(DistributionConfig, frozen=True):
    """OMGSR mixed Gaussian/generalized-Gaussian/plateau kernels."""

    sigma: Range = (0.2, 3.0)
    probabilities: Annotated[tuple[float, float, float, float, float, float], AfterValidator(_probabilities)] = (
        0.45,
        0.25,
        0.12,
        0.03,
        0.12,
        0.03,
    )
    generalized_beta: Range = (0.5, 4.0)
    plateau_beta: Range = (1.0, 2.0)

    @model_validator(mode="after")
    def validate_positive(self) -> Self:
        if min(self.sigma) <= 0 or not 0 < self.generalized_beta[0] <= 1 <= self.generalized_beta[1]:
            raise ValueError("Blur sigma must be positive and generalized beta must bracket one")
        if not 0 < self.plateau_beta[0] <= 1 <= self.plateau_beta[1]:
            raise ValueError("Plateau beta must be positive and bracket one")
        return self


class NoiseConfig(DistributionConfig, frozen=True):
    """Exactly one Gaussian, Poisson or camera-ISO draw per stage."""

    gaussian_probability: Probability = 1 / 3
    iso_probability: Probability = 1 / 3
    gaussian_sigma: Range = (1.0, 15.0)
    poisson_scale: Range = (0.05, 2.0)
    iso_shot_noise: Range = (0.0008, 0.015)
    iso_read_noise: Range = (0.0001, 0.006)
    iso_color_gain: Range = (0.92, 1.15)
    gray_probability: Probability = 0.4

    @model_validator(mode="after")
    def validate_noise_probabilities(self) -> Self:
        if self.gaussian_probability + self.iso_probability > 1:
            raise ValueError("Gaussian and ISO probabilities must sum to at most one")
        return self


class CompressionConfig(DistributionConfig, frozen=True):
    """Random DiffJPEG or WebP compression parameters."""

    jpeg_probability: Probability = 0.5
    jpeg_quality: Range = (60.0, 95.0)
    webp_quality: Range = (60.0, 95.0)

    @model_validator(mode="after")
    def validate_quality(self) -> Self:
        ranges = (self.jpeg_quality, self.webp_quality)
        if any(not 0 < bounds[0] < 100 or bounds[1] > 100 for bounds in ranges):
            raise ValueError("Compression quality bounds must lie within (0,100]")
        return self


class StageConfig(DistributionConfig, frozen=True):
    """First OMGSR degradation stage."""

    blur: BlurConfig = Field(default_factory=BlurConfig)
    sinc_probability: Probability = 0.1
    resize_probabilities: Probabilities = (0.2, 0.7, 0.1)
    resize_range: Range = (0.3, 1.5)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)

    @model_validator(mode="after")
    def validate_resize(self) -> Self:
        if not 0 < self.resize_range[0] <= 1 <= self.resize_range[1]:
            raise ValueError("Resize range must be positive and bracket one")
        return self


class SecondNoiseConfig(NoiseConfig, frozen=True):
    gaussian_sigma: Range = (1.0, 12.0)
    poisson_scale: Range = (0.05, 1.0)
    iso_shot_noise: Range = (0.001, 0.02)
    iso_read_noise: Range = (0.0002, 0.008)
    iso_color_gain: Range = (0.9, 1.2)


class SecondCompressionConfig(CompressionConfig, frozen=True):
    jpeg_quality: Range = (60.0, 100.0)
    webp_quality: Range = (60.0, 100.0)


class SecondBlurConfig(BlurConfig, frozen=True):
    sigma: Range = (0.2, 1.5)


class SecondStageConfig(StageConfig, frozen=True):
    """Second OMGSR degradation stage."""

    blur: SecondBlurConfig = Field(default_factory=SecondBlurConfig)
    blur_probability: Probability = 0.5
    resize_probabilities: Probabilities = (0.3, 0.4, 0.3)
    resize_range: Range = (0.8, 1.2)
    noise: SecondNoiseConfig = Field(default_factory=SecondNoiseConfig)
    compression: SecondCompressionConfig = Field(default_factory=SecondCompressionConfig)


class DegradationConfig(DistributionConfig, frozen=True):
    """OMGSR-F-1024 degradation with a fixed or continuously sampled scale."""

    enabled: bool = False
    scale: Scale = 4.0
    first: StageConfig = Field(default_factory=StageConfig)
    second: SecondStageConfig = Field(default_factory=SecondStageConfig)
    final_sinc_probability: Probability = 0.8
    gray_probability: Probability = 0.0
    color_jitter_probability: Probability = 0.0
    brightness: Range = (0.5, 1.5)
    contrast: Range = (0.5, 1.5)
    saturation: Range = (0.0, 1.5)
    hue: tuple[float, float] = (-0.1, 0.1)

    @model_validator(mode="after")
    def validate_hue(self) -> Self:
        if not -0.5 <= self.hue[0] <= self.hue[1] <= 0.5:
            raise ValueError("Hue range must be ordered within [-0.5,0.5]")
        return self
