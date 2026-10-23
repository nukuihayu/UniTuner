"""Declare optional update-boundary previews independently of numerical training choices."""

from pydantic import BaseModel, ConfigDict, Field

from unitunner.config.sampling import SamplingConfig
from unitunner.models.minimax_h3.config.inputs import MiniMaxH3SampleConfig
from unitunner.models.wan.config.sampling import WanSamplingConfig


class PreviewConfig(BaseModel):
    """Configure cadence and either flow prompts or restoration comparisons.

    Flow previews use samples. Restoration previews rotate through unique dataset
    items in a seed-fixed order, displaying low/restored/high comparisons. Settings
    do not affect resume identity or advance the training data cursor.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    every_updates: int = Field(default=100, gt=0)
    at_start: bool = False
    samples: tuple[SamplingConfig, ...] = ()
    videos: tuple[WanSamplingConfig, ...] = ()
    audio_video: tuple[MiniMaxH3SampleConfig, ...] = ()
    num_images: int = Field(default=1, gt=0)
