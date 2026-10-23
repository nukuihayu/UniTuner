"""Declare adapter algorithm choices independently of Torch and target discovery."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AdapterConfig(BaseModel):
    """Describe low-rank numerical choices; model bindings own target projections."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    algorithm: Literal["lora", "loha", "lokr"] = "lora"
    fused: bool = Field(
        default=True, description="Fuse eligible LoRA residuals and expose joint computation to the compiler"
    )
    rank: int = Field(default=32, gt=0)
    alpha: float = Field(default=32.0, gt=0)
