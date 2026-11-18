"""Declare AdamW numerical settings and implementation selection without Torch."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

AdamBeta = Annotated[float, Field(ge=0, lt=1)]


class AdamWConfig(BaseModel):
    """Share optimizer options across image, video, audio and restoration phases."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    betas: tuple[AdamBeta, AdamBeta] = Field(
        default=(0.9, 0.999), description="AdamW first/second moment decay factors, each in [0,1)"
    )
    eps: float = Field(default=1e-8, gt=0)
    backend: Literal["auto", "fused", "foreach", "single"] = "auto"
    implementation: Literal["torch", "adv_optm"] = "torch"
    state_precision: Literal["fp32", "bf16_sr", "int8_sr"] = "fp32"
    factored_2nd: bool = False

    @model_validator(mode="after")
    def validate_implementation(self) -> Self:
        if self.implementation == "torch" and (self.state_precision != "fp32" or self.factored_2nd):
            raise ValueError("Compressed optimizer states require implementation='adv_optm'")
        if self.implementation == "adv_optm" and self.backend != "auto":
            raise ValueError("adv_optm requires backend='auto'; PyTorch backend choices do not apply")
        return self


DEFAULT_ADAMW_CONFIG = AdamWConfig()
