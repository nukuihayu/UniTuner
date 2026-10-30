"""Declare FLUX.2 Dev/Klein topology separately from checkpoint training variants."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Flux2Config(BaseModel):
    """Pin bias-free SwiGLU streams, shared modulation and four-axis positions.

    Dev consumes embedded guidance; Klein base and step-distilled checkpoints
    share parameter geometry but use different sampling contracts. Reduced
    dimensions are explicit test topologies, never inferred from partial weights.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    version: Literal["dev", "klein-4b", "klein-9b"] = "dev"
    variant: Literal["base", "distilled"] = "distilled"
    in_channels: int = Field(default=128, gt=0)
    context_in_dim: int = Field(default=15360, gt=0)
    hidden_size: int = Field(default=6144, gt=0)
    num_heads: int = Field(default=48, gt=0)
    depth: int = Field(default=8, gt=0)
    depth_single_blocks: int = Field(default=48, gt=0)
    axes_dim: tuple[int, int, int, int] = (32, 32, 32, 32)
    theta: int = Field(default=2000, gt=0)
    mlp_ratio: float = Field(default=3.0, gt=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Reject invalid head/rotary geometry and unpublished Dev base selection."""
        if self.version == "dev" and self.variant != "distilled":
            raise ValueError("FLUX.2 Dev only publishes its guidance-distilled variant")
        if self.hidden_size % self.num_heads or sum(self.axes_dim) != self.hidden_size // self.num_heads:
            raise ValueError("FLUX.2 rotary axes must sum to the integral head width")
        if any(axis < 2 or axis % 2 for axis in self.axes_dim) or int(self.hidden_size * self.mlp_ratio) < 1:
            raise ValueError("FLUX.2 rotary axes must be positive/even and FFN width positive")
        return self

    @property
    def use_guidance_embed(self) -> bool:
        """Identify the published architecture's embedded guidance branch."""
        return self.version == "dev"


def flux2_config(
    version: Literal["dev", "klein-4b", "klein-9b"], *, variant: Literal["base", "distilled"] = "distilled"
) -> Flux2Config:
    """Resolve published dimensions without conflating Klein base and distilled."""
    if version == "klein-4b":
        return Flux2Config(
            version=version,
            variant=variant,
            context_in_dim=7680,
            hidden_size=3072,
            num_heads=24,
            depth=5,
            depth_single_blocks=20,
        )
    if version == "klein-9b":
        return Flux2Config(
            version=version,
            variant=variant,
            context_in_dim=12288,
            hidden_size=4096,
            num_heads=32,
            depth_single_blocks=24,
        )
    return Flux2Config(version=version, variant=variant)
