"""Declare the Wan CLIP visual topology and optional full-artifact text inventory."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WanClipConfig(BaseModel):
    """Describe ViT-H/14 with native token pooling and penultimate-layer conditions."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    image_size: int = Field(default=224, gt=0)
    patch_size: int = Field(default=14, gt=0)
    hidden_size: int = Field(default=1280, gt=0)
    output_dim: int = Field(default=1024, gt=0)
    num_heads: int = Field(default=16, gt=0)
    num_layers: int = Field(default=32, ge=2)
    mlp_ratio: int = Field(default=4, gt=0)
    norm_eps: float = Field(default=1e-5, gt=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Require complete square patches and integral attention heads."""
        if self.image_size % self.patch_size or self.hidden_size % self.num_heads:
            raise ValueError("Wan CLIP requires image_size divisible by patch_size and hidden_size by num_heads")
        return self


class WanClipTextArtifactConfig(BaseModel):
    """Describe unused XLM-Roberta keys for strict full-artifact validation only.

    No text encoder is constructed or executed. Published defaults describe the
    Wan XLM-Roberta CLIP checkpoint; explicit reduced fixtures are separate.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    vocab_size: int = Field(default=250002, gt=0)
    max_length: int = Field(default=514, gt=0)
    type_size: int = Field(default=1, gt=0)
    hidden_size: int = Field(default=1024, gt=0)
    num_layers: int = Field(default=24, gt=0)
