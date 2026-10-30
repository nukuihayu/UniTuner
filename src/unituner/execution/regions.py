"""Describe immutable logical execution regions independently of their installations."""

from dataclasses import dataclass
from typing import Literal

from torch import nn

type RegionBranch = Literal["visual", "condition", "text", "audio", "refiner", "joint"]


@dataclass(frozen=True, slots=True)
class ModelRegion:
    """Identify a concrete invocation's branch and logical index, even for reused blocks."""

    module: nn.Module
    logical_index: int
    branch: RegionBranch

    def __post_init__(self) -> None:
        if self.logical_index < 0:
            raise ValueError("Region logical_index must be nonnegative")


@dataclass(frozen=True, slots=True)
class ModelRegions:
    """Keep call geometry separate from module attributes and resource ownership."""

    regions: tuple[ModelRegion, ...]

    def __post_init__(self) -> None:
        identities = [(region.branch, region.logical_index) for region in self.regions]
        if len(identities) != len(set(identities)):
            raise ValueError("Region branch/logical_index identities must be unique")
