"""Declare AdamW numerical settings and implementation selection without Torch."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

AdamBeta = Annotated[float, Field(ge=0, lt=1)]


class AdamWConfig(BaseModel):
    """Share optimizer options across image, video, audio and restoration phases."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    betas: tuple[AdamBeta, AdamBeta] = Field(
        default=(0.9, 0.999), description="AdamW first/second moment decay factors, each in [0,1)"
    )
    eps: float = Field(default=1e-8, gt=0)
    backend: Literal["auto", "fused", "foreach", "single"] = "auto"


DEFAULT_ADAMW_CONFIG = AdamWConfig()
