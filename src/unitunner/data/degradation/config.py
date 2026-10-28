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
    """Gaussian/generalized/plateau, each isotropic then anisotropic; optional sinc.

    First-stage sigma uses a 1024-pixel short-edge reference with square-root
    scaling. Second-stage sigma is measured directly in low-resolution pixels.
    """

    sigma: Range = (0.4, 2.8)
    probabilities: Annotated[tuple[float, float, float, float, float, float], AfterValidator(_probabilities)] = (
        0.50,
        0.25,
        0.10,
        0.05,
        0.07,
        0.03,
    )
    generalized_beta: Range = (0.5, 4.0)
    plateau_beta: Range = (1.0, 2.0)
    sinc_probability: Probability = 0.05
    motion_probability: Probability = 0.12
    defocus_probability: Probability = 0.10

    @model_validator(mode="after")
    def validate_positive(self) -> Self:
        if min(self.sigma) <= 0 or not 0 < self.generalized_beta[0] <= 1 <= self.generalized_beta[1]:
            raise ValueError("Blur sigma must be positive and generalized beta must bracket one")
        if not 0 < self.plateau_beta[0] <= 1 <= self.plateau_beta[1]:
            raise ValueError("Plateau beta must be positive and bracket one")
        if self.sinc_probability + self.motion_probability + self.defocus_probability > 1:
            raise ValueError("Sinc, motion and defocus probabilities must sum to at most one")
        return self


class NoiseConfig(DistributionConfig, frozen=True):
    """Gaussian/Poisson/ISO probabilities; sigma is in 8-bit intensity units."""

    probabilities: Probabilities = (0.35, 0.15, 0.50)
    gaussian_sigma: Range = (1.0, 9.0)
    poisson_scale: Range = (0.05, 0.5)
    iso_shot: Range = (0.0002, 0.006)
    iso_read: Range = (0.000005, 0.00015)
    exposure: Range = (0.75, 1.0)
    iso_color_gain: Range = (0.92, 1.15)
    gray_probability: Probability = 0.25

    @model_validator(mode="after")
    def validate_exposure(self) -> Self:
        if not 0 < self.exposure[0] <= self.exposure[1] <= 1:
            raise ValueError("Noise exposure must be within (0,1]")
        return self


class CompressionConfig(DistributionConfig, frozen=True):
    """JPEG/WebP/JPEG2000 probabilities and codec-native quality ranges."""

    probabilities: Probabilities = (0.85, 0.15, 0.0)
    recompression_probability: Probability = 0.25
    jpeg_quality: Range = (45.0, 95.0)
    webp_quality: Range = (65.0, 98.0)
    jpeg2000_quality: Range = (300.0, 900.0)

    @model_validator(mode="after")
    def validate_quality(self) -> Self:
        if self.jpeg_quality[1] > 100 or self.webp_quality[1] > 100 or self.jpeg2000_quality[1] > 1000:
            raise ValueError("JPEG/WebP quality must be <=100; JPEG2000 uses OpenCV's 0..1000 scale")
        return self


class StageConfig(DistributionConfig, frozen=True):
    """Ordered acquisition stage; optional intermediate encoding for web reposts."""

    blur: BlurConfig = Field(default_factory=BlurConfig)
    blur_probability: Probability = 0.8
    resize_probabilities: Probabilities = (0.0, 0.6, 0.4)
    resize_range: Range = (0.75, 1.0)
    noise_probability: Probability = 0.8
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
    probabilities: Probabilities = (0.7, 0.1, 0.2)
    gaussian_sigma: Range = (0.2, 2.0)
    poisson_scale: Range = (0.02, 0.15)
    iso_shot: Range = (0.00001, 0.0002)
    iso_read: Range = (0.0000001, 0.000004)
    iso_color_gain: Range = (0.95, 1.05)


class SecondCompressionConfig(CompressionConfig, frozen=True):
    """Second-stage codec distribution with independently overridable fields."""

    probabilities: Probabilities = (0.8, 0.2, 0.0)
    jpeg_quality: Range = (35.0, 95.0)
    webp_quality: Range = (50.0, 98.0)
    jpeg2000_quality: Range = (250.0, 900.0)


class SecondBlurConfig(BlurConfig, frozen=True):
    """Second-stage blur range; other kernel distributions match the first stage."""

    sigma: Range = (0.15, 0.65)
    sinc_probability: Probability = 0.03
    motion_probability: Probability = 0.0
    defocus_probability: Probability = 0.0


class SecondStageConfig(StageConfig, frozen=True):
    """Second-stage defaults also apply when TOML overrides only one nested field."""

    blur: SecondBlurConfig = Field(default_factory=SecondBlurConfig)
    blur_probability: Probability = 0.15
    resize_probabilities: Probabilities = (0.1, 0.2, 0.7)
    resize_range: Range = (0.9, 1.1)
    noise_probability: Probability = 0.1
    compression_probability: Probability = 0.9
    noise: SecondNoiseConfig = Field(default_factory=SecondNoiseConfig)
    compression: SecondCompressionConfig = Field(default_factory=SecondCompressionConfig)


class DegradationConfig(DistributionConfig, frozen=True):
    """Dataset-owned super-resolution acquisition and delivery distributions.

    First-stage blur scales by sqrt(short edge / 1024), capped to [0.5, 1.5].
    Second-stage blur uses local pixels. Scene probabilities: photo/web/low-light.
    Sinc cutoff and support remain in the local pixel grid, independent of HQ size.
    Color jitter and grayscale affect LQ only; HQ remains an unchanged target.
    """

    enabled: bool = False
    scales: tuple[Annotated[int, Field(ge=1, le=16)], ...] = (2, 3, 4)
    scenario_probabilities: Probabilities = (0.25, 0.35, 0.40)
    processing_probabilities: Probabilities = (0.30, 0.45, 0.25)
    grain_smear_probability: Probability = 0.45
    compression_blur_probability: Probability = 0.25
    grain_strength: Range = (0.75, 1.5)
    smear_strength: Range = (0.5, 1.0)
    smear_detail_protection: Probability = 0.6
    resize_modes: tuple[Literal["area", "bilinear", "bicubic", "nearest"], ...] = (
        "area",
        "bilinear",
        "bicubic",
    )
    first: StageConfig = Field(default_factory=StageConfig)
    second: SecondStageConfig = Field(default_factory=SecondStageConfig)
    final_sinc_probability: Probability = 0.1
    gray_probability: Probability = 0.0
    color_jitter_probability: Probability = 0.0
    brightness: Range = (0.5, 1.5)
    contrast: Range = (0.5, 1.5)
    saturation: Range = (0.0, 1.5)
    hue: tuple[float, float] = (-0.1, 0.1)

    @model_validator(mode="after")
    def validate_choices(self) -> Self:
        if self.grain_strength[1] > 4 or self.smear_strength[1] > 1:
            raise ValueError("grain_strength must be within [0,4] and smear_strength within [0,1]")
        if not self.scales or not self.resize_modes:
            raise ValueError("Degradation scales and resize modes cannot be empty")
        if not -0.5 <= self.hue[0] <= self.hue[1] <= 0.5:
            raise ValueError("Hue range must be ordered within [-0.5,0.5]")
        return self
