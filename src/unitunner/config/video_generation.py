"""Select a strict native video-generation contract without importing runtimes."""

from pydantic import ConfigDict, RootModel

from unitunner.models.minimax_h3.config.generation import MiniMaxH3GenerationConfig
from unitunner.models.wan.config.generation import WanGenerationConfig


class VideoGenerationConfig(RootModel[WanGenerationConfig | MiniMaxH3GenerationConfig]):
    """Validate the family-owned Wan or MiniMaxH3 generation document."""

    model_config = ConfigDict(frozen=True, strict=True)
