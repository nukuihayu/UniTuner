"""Immutable dataset-owned two-stage degradation distributions.

Ranges use inclusive bounds;
probability tuples follow the explicit orders documented on their fields.
"""

from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def _range(value: tuple[float, float]) -> tuple[float, float]:
    if value[0] < 0 or value[0] > value[1]:
        raise ValueError("Expected a nonnegative ordered range")
    return value


def _probabilities(value: tuple[float, ...]) -> tuple[float, ...]:
    if min(value) < 0 or abs(sum(value) - 1) > 1e-12:
        raise ValueError("Probabilities must be nonnegative and sum to one")
    return value


type Range = Annotated[tuple[float, float], AfterValidator(_range)]
type Probabilities = Annotated[tuple[float, float, float], AfterValidator(_probabilities)]
type Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class DistributionConfig(BaseModel, frozen=True):
    """Strict immutable input; NaN, infinity and unknown distribution keys fail."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)


class BlurConfig(DistributionConfig, frozen=True):
    """Gaussian/generalized/plateau, handheld motion and defocus PSFs.

    All spatial ranges use a 1024-pixel layer-input short-edge reference and
    gentle fourth-root scaling below 1024 and square-root scaling above it,
    capped to [0.5, 2.0].
    """

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
    motion_probability: Probability = 0.20
    defocus_probability: Probability = 0.15
    motion_length: Range = (0.0, 10.0)
    defocus_radius: Range = (0.5, 4.0)

    @model_validator(mode="after")
    def validate_positive(self) -> Self:
        if min(self.sigma) <= 0 or not 0 < self.generalized_beta[0] <= 1 <= self.generalized_beta[1]:
            raise ValueError("Blur sigma must be positive and generalized beta must bracket one")
        if not 0 < self.plateau_beta[0] <= 1 <= self.plateau_beta[1]:
            raise ValueError("Plateau beta must be positive and bracket one")
        if self.motion_length[1] > 64 or self.defocus_radius[1] > 16:
            raise ValueError("motion_length must be <=64 and defocus_radius <=16")
        if self.motion_probability + self.defocus_probability > 1:
            raise ValueError("Motion and defocus probabilities must sum to at most one")
        return self


class NoiseConfig(DistributionConfig, frozen=True):
    """Gaussian/Poisson/ISO probabilities; sigma is in 8-bit intensity units."""

    probabilities: Probabilities = (0.50, 0.50, 0.0)
    gaussian_sigma: Range = (1.0, 30.0)
    poisson_scale: Range = (0.05, 3.0)
    iso_shot: Range = (0.0003, 0.009)
    iso_read: Range = (0.000008, 0.00025)
    exposure: Range = (0.75, 1.0)
    iso_color_gain: Range = (0.92, 1.15)
    gray_probability: Probability = 0.40

    @model_validator(mode="after")
    def validate_exposure(self) -> Self:
        if not 0 < self.exposure[0] <= self.exposure[1] <= 1:
            raise ValueError("Noise exposure must be within (0,1]")
        return self


class CompressionConfig(DistributionConfig, frozen=True):
    """JPEG/WebP/JPEG2000 probabilities and codec-native quality ranges."""

    probabilities: Probabilities = (0.85, 0.15, 0.0)
    recompression_probability: Probability = 0.25
    jpeg_quality: Range = (30.0, 95.0)
    webp_quality: Range = (45.0, 98.0)
    jpeg2000_quality: Range = (250.0, 900.0)

    @model_validator(mode="after")
    def validate_quality(self) -> Self:
        if self.jpeg_quality[1] > 100 or self.webp_quality[1] > 100 or self.jpeg2000_quality[1] > 1000:
            raise ValueError("JPEG/WebP quality must be <=100; JPEG2000 uses OpenCV's 0..1000 scale")
        return self


class StageConfig(DistributionConfig, frozen=True):
    """Ordered acquisition stage; optional intermediate encoding for web reposts."""

    blur: BlurConfig = Field(default_factory=BlurConfig)
    blur_probability: Probability = 1.0
    resize_probabilities: Probabilities = (0.2, 0.7, 0.1)
    resize_range: Range = (0.15, 1.5)
    noise_probability: Probability = 1.0
    compression_probability: Probability = 0.75
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)

    @model_validator(mode="after")
    def validate_resize(self) -> Self:
        if not 0 < self.resize_range[0] <= 1 <= self.resize_range[1]:
            raise ValueError("Resize range must be positive and bracket one")
        return self


class SecondNoiseConfig(NoiseConfig, frozen=True):
    """Second-stage camera distribution with independently overridable fields."""

    exposure: Range = (1.0, 1.0)
    probabilities: Probabilities = (0.50, 0.50, 0.0)
    gaussian_sigma: Range = (1.0, 25.0)
    poisson_scale: Range = (0.05, 2.5)
    iso_shot: Range = (0.00002, 0.0004)
    iso_read: Range = (0.0000002, 0.000008)
    iso_color_gain: Range = (0.95, 1.05)


class SecondCompressionConfig(CompressionConfig, frozen=True):
    """Second-stage codec distribution with independently overridable fields."""

    probabilities: Probabilities = (0.8, 0.2, 0.0)
    jpeg_quality: Range = (30.0, 95.0)
    webp_quality: Range = (45.0, 98.0)
    jpeg2000_quality: Range = (200.0, 900.0)


class SecondBlurConfig(BlurConfig, frozen=True):
    """Milder second-stage blur ranges with the same resolution adaptation."""

    sigma: Range = (0.2, 1.5)
    motion_probability: Probability = 0.0
    defocus_probability: Probability = 0.0


class SecondStageConfig(StageConfig, frozen=True):
    """Second-stage defaults also apply when TOML overrides only one nested field."""

    blur: SecondBlurConfig = Field(default_factory=SecondBlurConfig)
    blur_probability: Probability = 0.8
    resize_probabilities: Probabilities = (0.3, 0.4, 0.3)
    resize_range: Range = (0.3, 1.2)
    noise_probability: Probability = 1.0
    compression_probability: Probability = 0.9
    noise: SecondNoiseConfig = Field(default_factory=SecondNoiseConfig)
    compression: SecondCompressionConfig = Field(default_factory=SecondCompressionConfig)


class DegradationConfig(DistributionConfig, frozen=True):
    """Dataset-owned super-resolution acquisition and delivery distributions.

    Spatial kernels use the current layer-input short edge with scaling from a
    1024-pixel reference, capped to [0.5, 2.0]. Scaling uses a fourth root below
    the reference and a square root above it. Scene
    probabilities are photo/web/low-light.
    Color jitter and grayscale affect LQ only; HQ remains an unchanged target.
    """

    enabled: bool = False
    scales: tuple[Annotated[float, Field(ge=1, le=16)], ...] | None = None
    scale_range: Range = (1.0, 8.0)
    scenario_probabilities: Probabilities = (0.25, 0.35, 0.40)
    noise_strength: Range = (1.0, 1.0)
    processing_deadzone: Probability = 0.1
    grain_strength: Range = (0.0, 1.5)
    smear_strength: Range = (0.0, 0.8)
    smear_detail_protection: Probability = 0.6
    grain_scale_range: Range = (1.0, 4.0)
    sharpen_strength: Range = (0.0, 0.4)
    resize_modes: tuple[Literal["area", "bilinear", "bicubic", "nearest"], ...] = (
        "area",
        "bilinear",
        "bicubic",
    )
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
    def validate_choices(self) -> Self:
        if self.noise_strength[1] > 4:
            raise ValueError("noise_strength must be within [0,4]")
        if self.grain_strength[1] > 4 or self.smear_strength[1] > 1:
            raise ValueError("grain_strength must be within [0,4] and smear_strength within [0,1]")
        if not 1 <= self.scale_range[0] <= self.scale_range[1] <= 16:
            raise ValueError("scale_range must be within [1,16]")
        if not 1 <= self.grain_scale_range[0] <= self.grain_scale_range[1] <= 16:
            raise ValueError("grain_scale_range must be within [1,16]")
        if self.sharpen_strength[1] > 1:
            raise ValueError("sharpen_strength must be within [0,1]")
        if self.scales == () or not self.resize_modes:
            raise ValueError("Degradation scales and resize modes cannot be empty")
        if not -0.5 <= self.hue[0] <= self.hue[1] <= 0.5:
            raise ValueError("Hue range must be ordered within [-0.5,0.5]")
        return self
