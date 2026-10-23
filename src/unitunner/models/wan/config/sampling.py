"""Declare native Wan video sampling choices independently of image-only controls."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WanSamplingConfig(BaseModel):
    """Select explicit output geometry, native solver/CFG and optional image/control sources."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    prompt: str = ""
    negative_prompt: str = ""
    width: int = Field(default=832, ge=16)
    height: int = Field(default=480, ge=16)
    frames: int = Field(default=81, gt=0)
    fps: int = Field(default=16, gt=0)
    steps: int = Field(default=40, gt=0)
    solver: Literal["euler", "dpm++", "unipc"] = "unipc"
    shift: float = Field(default=5.0, gt=0)
    guidance: float = Field(default=5.0, ge=0)
    seed: int = 42
    attention_backend: Literal["torch", "kitchen-int8"] = "torch"
    conditioning_image: str | None = Field(default=None, min_length=1)
    control_video: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Require16-aligned pixels and native1+4k temporal frames without truncation."""
        if self.width % 16 or self.height % 16 or (self.frames - 1) % 4:
            raise ValueError("Wan sampling requires16-aligned width/height and1+4k frames")
        return self


def validate_wan_sampling(sampling: WanSamplingConfig, *, version: str | None, variant: str) -> None:
    """Reject unsupported variant/conditioning combinations before asset IO."""
    if version not in ("2.1", "2.2") or variant not in ("t2v", "i2v"):
        raise ValueError("Wan video sampling requires2.1/2.2 t2v/i2v")
    if (sampling.conditioning_image is not None) != (variant == "i2v"):
        raise ValueError("Wan conditioning_image is required exactly for I2V")
