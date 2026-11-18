"""Declare optional DWT-domain supervision using normalized noise fractions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WaveletLossConfig(BaseModel):
    """Weight DWT band MSE; zero boundaries and upstream default band coefficients."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    weight: float = Field(default=1.0, gt=0)
    wavelet: Literal["haar", "db4"] = "db4"
    levels: int = Field(default=3, ge=1, le=8)
    ll: float = Field(default=0.1, ge=0)
    lh: float = Field(default=0.01, ge=0)
    hl: float = Field(default=0.01, ge=0)
    hh: float = Field(default=0.05, ge=0)
    ll_level_threshold: int | None = -1
    normalize_bands: bool = False
    timestep_cutoff: float = Field(default=0.7, ge=0, le=1)
    timestep_transition_width: float = Field(default=0.4, gt=0)
