"""Keep restoration coefficients and method selection explicit."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from unitunner.config.diagnostics import ConfigRuleError
from unitunner.config.features import LayerSyncConfig, RepaConfig
from unitunner.config.routing import TreadConfig
from unitunner.config.wavelet import WaveletLossConfig
from unitunner.models.minimax_h3.config.training import MiniMaxH3FlowConfig
from unitunner.models.qwen_image.config.training import QwenImageFlowConfig


class RestorationLossWeights(BaseModel):
    """Preserve original coefficients using canonical lowercase loss identifiers."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    lrr: float = Field(default=5.0, ge=0)
    dv3d: float = Field(default=3.0, ge=0)
    l1: float = Field(default=0.5, ge=0)
    gan: float = Field(default=0.5, ge=0)
    fdl: float = Field(default=0.5, ge=0)
    histogram: float = Field(default=0.0, ge=0)
    phase: float = Field(default=0.0, ge=0)
    ms_ssim: float = Field(default=0.0, ge=0)


class RestorationConfig(BaseModel):
    """Select restoration time, posterior policy and external feature artifacts.

    Omitted posterior retains Trainer's Z-Image/FLUX sampled policy and native
    Qwen/Krea2 sampling. Calibration uses the same policy with isolated seeded
    RNG; previews and inference use means.
    Midtimestep defaults to 244 as a starting choice, not a universally optimal
    sigma. Explicit sigma or SNR calibration select an alternative time policy.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    midtimestep: int = Field(default=244, ge=1, le=998)
    sigma: float | None = Field(default=None, gt=0, lt=1, allow_inf_nan=False)
    calibrate: bool = False
    calibration_pairs: int = Field(default=32, gt=0)
    calibration_sampling: Literal["random", "tiles"] = "random"
    encoder_rank: int = Field(default=4, ge=0)
    posterior: Literal["mean", "sample"] | None = None
    histogram_bins: int = Field(default=32, ge=2)
    phase_period: int = Field(default=16, ge=2)
    phase_cycles: int = Field(default=8, ge=2)
    phase_epsilon: float = Field(default=1e-3, ge=1e-12, le=1.0, allow_inf_nan=False)
    dino_weights: str | None = Field(
        default=None,
        description="Local DINO ConvNeXt checkpoint (.pt/.pth/.safetensors); required when losses.dv3d or losses.gan is positive",
    )
    vgg_weights: str | None = Field(
        default=None, description="Local VGG19 checkpoint (.pt/.pth/.safetensors); required when losses.fdl is positive"
    )
    feature_precision: Literal["bf16", "fp32"] = "bf16"
    dino_widths: tuple[PositiveInt, ...] = Field(
        default=(192, 384, 768, 1536),
        min_length=4,
        max_length=4,
        description="Four DINO ConvNeXt stage widths; defaults match ConvNeXt Large",
    )
    dino_depths: tuple[PositiveInt, ...] = Field(default=(3, 3, 27, 3), min_length=4, max_length=4)
    vgg_widths: tuple[PositiveInt, ...] = Field(default=(64, 128, 256, 512, 512), min_length=5, max_length=5)

    @model_validator(mode="after")
    def validate_calibration(self) -> Self:
        """Reject contradictory restoration calibration settings."""
        if self.calibrate and self.sigma is not None:
            raise ConfigRuleError(
                "sigma",
                "calibrate and explicit sigma are mutually exclusive",
                "Remove sigma to estimate it with calibrate=true, or set calibrate=false to use explicit sigma.",
            )
        if not self.calibrate and self.calibration_sampling != "random":
            raise ConfigRuleError(
                "calibration_sampling",
                "Tile calibration requires calibrate = true",
                "Enable calibrate=true or use the default random sampling.",
            )
        if not self.calibrate and self.calibration_pairs != 32:
            raise ConfigRuleError(
                "calibration_pairs",
                "Nondefault calibration_pairs requires calibrate = true",
                "Enable calibrate=true or leave calibration_pairs at its default of 32.",
            )
        return self


class MethodConfig(BaseModel):
    """Reject method settings that would otherwise be silently ignored."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    kind: Literal["flow-matching", "restoration"] = "flow-matching"
    layersync: LayerSyncConfig | None = None
    repa: RepaConfig | None = None
    tread: TreadConfig | None = None
    wavelet: WaveletLossConfig | None = None
    guidance: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    restoration: RestorationConfig | None = None
    qwen_image: QwenImageFlowConfig | None = None
    minimax_h3: MiniMaxH3FlowConfig | None = None

    @model_validator(mode="after")
    def validate_method(self) -> Self:
        """Require restoration settings exactly when that method is selected."""
        if self.kind == "restoration" and self.repa is not None:
            raise ConfigRuleError(
                "repa",
                "REPA currently requires flow-matching",
                "Remove method.repa for restoration, or set method.kind='flow-matching'.",
            )
        if self.kind == "restoration" and self.tread is not None:
            raise ConfigRuleError(
                "tread",
                "TREAD currently requires flow-matching",
                "Remove method.tread for restoration, or set method.kind='flow-matching'.",
            )
        if self.kind == "restoration" and self.layersync is not None:
            raise ConfigRuleError(
                "layersync",
                "LayerSync currently requires flow-matching",
                "Remove method.layersync for restoration, or set method.kind='flow-matching'.",
            )
        if self.kind == "restoration" and self.guidance is not None:
            raise ConfigRuleError(
                "guidance",
                "method.guidance requires flow-matching",
                "Remove method.guidance for restoration, or set method.kind='flow-matching'.",
            )
        if (self.kind == "restoration") != (self.restoration is not None):
            raise ConfigRuleError(
                "restoration",
                "restoration settings must be supplied exactly for the restoration method",
                "Supply [method.restoration] when method.kind='restoration'; remove it for flow-matching.",
            )
        if self.minimax_h3 is not None and self.kind != "flow-matching":
            raise ConfigRuleError(
                "minimax_h3",
                "method.minimax_h3 requires flow-matching",
                "Remove method.minimax_h3 or select flow-matching.",
            )
        if self.qwen_image is not None and self.kind != "flow-matching":
            raise ConfigRuleError(
                "qwen_image",
                "method.qwen_image requires flow-matching",
                "Remove method.qwen_image or select flow-matching.",
            )
        return self
