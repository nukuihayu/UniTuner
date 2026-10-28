"""Declare execution choices without probing devices or importing Torch."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unitunner.artifacts.formats.precision import TrainingPrecision
from unitunner.artifacts.formats.quantization import (
    DEFAULT_GROUP_SIZE,
    DEFAULT_ROTATION_SIZE,
    FORMATS,
    QUANTIZATION_ALIASES,
    WeightQuantization,
    canonical_quantization,
)
from unitunner.config.diagnostics import ConfigRuleError


class ExecutionConfig(BaseModel):
    """Separate user policy from a resolved execution installation."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    precision: TrainingPrecision = Field(
        default="auto",
        description="Compute/storage precision: auto follows base weights; bf16/fp16/fp32 or supported Kitchen formats; offline-calibrated formats are not accepted",
    )
    fp16_loss_scale: float = Field(default=128.0, ge=1.0, allow_inf_nan=False)
    compile: bool | Literal["auto"] = Field(
        default=False, description="Native eager by default; true or auto opts into CUDA Inductor"
    )
    recompute: Literal["none", "block"] = "none"
    recompute_interval: int = Field(default=1, gt=0)
    restoration_recompute: Literal["inherit", "none", "block"] = Field(
        default="inherit", description="VAE and DINO recomputation; inherit follows the main model policy"
    )
    attention_activation_offload: bool = False
    residency: Literal["resident", "frozen-blocks", "vbar", "trainable-blocks"] = "resident"
    use_pinned_memory_for_block_swap: bool | None = None
    frozen_weight_cache_mb: int | None = Field(
        default=None, ge=0, description="Automatic frozen CUDA cache, or retained MiB plus one current-block workspace"
    )
    quantization: WeightQuantization = "none"
    quantization_rotation_size: int = Field(default=DEFAULT_ROTATION_SIZE, ge=4)
    quantization_group_size: int = Field(default=DEFAULT_GROUP_SIZE, gt=0)

    @property
    def offloads_frozen_weights(self) -> bool:
        """Identify policies that keep frozen master weights outside the active device."""
        return self.residency in {"frozen-blocks", "vbar"}

    @model_validator(mode="before")
    @classmethod
    def resolve_weight_precision(cls, value: object) -> object:
        """Accept Kitchen formats through the same precision input as floating types."""
        if not isinstance(value, dict):
            return value
        for field in ("precision", "quantization"):
            requested_format = value.get(field)
            if (
                isinstance(requested_format, str)
                and requested_format in FORMATS
                and FORMATS[requested_format].calibrated
            ):
                raise ConfigRuleError(
                    field,
                    f"execution.{field} does not accept offline-calibrated format {requested_format}",
                    "Use bf16/fp16/fp32/auto or an online Kitchen format; offline calibration is not supported at this entry point.",
                )
        precision = value.get("precision")
        if precision == "keep":
            return {**value, "precision": "auto"}
        if isinstance(precision, str) and precision in (*FORMATS, *QUANTIZATION_ALIASES):
            requested = canonical_quantization(precision)
            quantization = value.get("quantization", "none")
            if not isinstance(quantization, str):
                raise ValueError("execution.quantization must name a Kitchen format or none")
            existing = canonical_quantization(quantization)
            if existing not in ("none", requested):
                raise ConfigRuleError(
                    "precision",
                    "execution.precision and execution.quantization select different Kitchen formats",
                    "Select the same Kitchen format in both fields, or specify only execution.precision.",
                )
            return {**value, "precision": "auto", "quantization": requested}
        return value

    @model_validator(mode="before")
    @classmethod
    def infer_required_recompute(cls, value: object) -> object:
        """Offloaded frozen blocks necessarily replay; infer this when omitted."""
        if isinstance(value, dict) and value.get("residency") in ("frozen-blocks", "vbar") and "recompute" not in value:
            return {**value, "recompute": "block"}
        return value

    @model_validator(mode="after")
    def validate_block_swap(self) -> Self:
        """Require a block residency policy for explicitly requested pinned masters."""
        if self.use_pinned_memory_for_block_swap and self.residency == "resident":
            raise ConfigRuleError(
                "use_pinned_memory_for_block_swap",
                "use_pinned_memory_for_block_swap requires block residency",
                "Disable pinned block swap or choose a supported block residency policy.",
            )
        if self.frozen_weight_cache_mb and self.residency != "frozen-blocks":
            raise ConfigRuleError(
                "frozen_weight_cache_mb",
                "frozen_weight_cache_mb requires frozen-blocks residency",
                "Remove frozen_weight_cache_mb or set execution.residency='frozen-blocks'.",
            )
        return self
