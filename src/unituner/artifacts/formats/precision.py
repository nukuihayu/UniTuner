"""Name floating precision without importing a compute provider."""

from typing import Literal

type FloatingPrecision = Literal["fp32", "fp16", "bf16"]
type TrainingPrecision = Literal["auto", "fp32", "fp16", "bf16"]
