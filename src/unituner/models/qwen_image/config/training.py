"""Declare native qwen image training policies."""

from pydantic import BaseModel, ConfigDict


class QwenImageFlowConfig(BaseModel):
    """Choose whether layered flow learns the composite as well as foregrounds."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    remove_base: bool = False
