"""Declare weight conversion formats without importing optional kernels."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from unituner.artifacts.formats.precision import FloatingPrecision

DEFAULT_ROTATION_SIZE = 256
DEFAULT_GROUP_SIZE = 16


@dataclass(frozen=True)
class TensorStorage:
    """Map a Kitchen parameter to its canonical tensor suffix and byte view."""

    parameter: str
    suffix: str
    byte_dtype: str | None = None
    optional: bool = False


_SCALE = TensorStorage("scale", "scale")
_ASYMMETRIC_TENSORS = (
    TensorStorage("scale", "s_rel", "float8_e4m3fn"),
    TensorStorage("s_channel", "s_channel"),
    TensorStorage("correction", "correction", optional=True),
    TensorStorage("codebook", "codebook", optional=True),
)


@dataclass(frozen=True)
class WeightFormat:
    """Describe a canonical layout, quantizer options and parameter storage."""

    layout: str
    wire_format: str
    options: tuple[tuple[str, object], ...] = ()
    calibrated: bool = False
    tensors: tuple[TensorStorage, ...] = (_SCALE,)
    uses_rotation: bool = False
    uses_group_size: bool = False

    def tensor_suffix(self, parameter: str) -> str:
        """Locate a parameter's sidecar, defaulting to Kitchen's field name."""
        return next((tensor.suffix for tensor in self.tensors if tensor.parameter == parameter), parameter)


FORMATS = {
    "float8_e4m3fn": WeightFormat("TensorCoreFP8Layout", "float8_e4m3fn"),
    "float8_e5m2": WeightFormat("TensorCoreFP8Layout", "float8_e5m2"),
    "mxfp8": WeightFormat(
        "TensorCoreMXFP8Layout", "mxfp8", tensors=(TensorStorage("scale", "scale", "float8_e8m0fnu"),)
    ),
    "nvfp4": WeightFormat(
        "TensorCoreNVFP4Layout",
        "nvfp4",
        tensors=(TensorStorage("scale", "scale_2"), TensorStorage("block_scale", "scale", "float8_e4m3fn")),
    ),
    "int8_tensorwise": WeightFormat("TensorWiseINT8Layout", "int8_tensorwise"),
    "int8_rowwise": WeightFormat("TensorWiseINT8Layout", "int8_tensorwise", (("per_channel", True),)),
    "int8_convrot": WeightFormat(
        "TensorWiseINT8Layout", "int8_tensorwise", (("per_channel", True), ("convrot", True)), uses_rotation=True
    ),
    "convrot_w4a4": WeightFormat("TensorCoreConvRotW4A4Layout", "convrot_w4a4", uses_rotation=True),
    "convrot_w4a8": WeightFormat(
        "TensorCoreConvRotW4A4Layout", "convrot_w4a4", (("linear_dtype", "int8"),), uses_rotation=True
    ),
    "asym_w4a8_int8": WeightFormat(
        "AsymW4A8Int8Layout", "asym_w4a8_int8", tensors=_ASYMMETRIC_TENSORS, uses_rotation=True, uses_group_size=True
    ),
    "w6a8_int8": WeightFormat(
        "AsymW4A8Int8Layout",
        "w6a8_int8",
        (("bits", 6), ("codebook", False)),
        tensors=_ASYMMETRIC_TENSORS,
        uses_rotation=True,
        uses_group_size=True,
    ),
    "awq_w4a16": WeightFormat(
        "TensorCoreAWQW4A16Layout", "awq_w4a16", calibrated=True, tensors=(_SCALE, TensorStorage("zeros", "zeros"))
    ),
    "svdquant_w4a4": WeightFormat(
        "TensorCoreSVDQuantW4A4Layout",
        "svdquant_w4a4",
        calibrated=True,
        tensors=(
            _SCALE,
            TensorStorage("proj_down", "proj_down"),
            TensorStorage("proj_up", "proj_up"),
            TensorStorage("smooth_factor", "smooth_factor"),
        ),
    ),
}

QUANTIZATION_ALIASES = {"fp8": "float8_e4m3fn", "convrot-int8": "int8_convrot"}


def canonical_quantization(name: str) -> str:
    """Resolve public aliases without changing stored configuration or validating it."""
    return QUANTIZATION_ALIASES.get(name, name)


# Shared by denoiser preparation and frozen text encoding; calibrated imports
# intentionally have no automatic quantization option.
WeightQuantization = Literal[
    "none",
    "fp8",
    "convrot-int8",
    "float8_e4m3fn",
    "float8_e5m2",
    "mxfp8",
    "nvfp4",
    "int8_tensorwise",
    "int8_rowwise",
    "int8_convrot",
    "convrot_w4a4",
    "convrot_w4a8",
    "asym_w4a8_int8",
    "w6a8_int8",
]


PRECISIONS = ("keep", "fp32", "fp16", "bf16", *FORMATS)


class QuantizationFormat(BaseModel):
    """Pin scale axes and algorithm identity at the external artifact boundary."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal[1] = 1
    algorithm: Literal["fp8-e4m3", "fp8-e5m2", "convrot-int8"]
    scale_key: str = Field(min_length=1)
    scale_axis: int | None
    original_dtype: FloatingPrecision
