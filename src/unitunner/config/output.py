"""Declare artifact publication and checkpoint output policy."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from unitunner.artifacts.formats.precision import FloatingPrecision


class OutputConfig(BaseModel):
    """Configure publication without affecting numerical identity."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    directory: str = Field(
        default="runs", min_length=1, description="Omit to use runs/<name>_<timestamp>; explicit paths are reused"
    )
    name: str = Field(default="model", min_length=1)
    log_every_seconds: float = Field(default=60.0, gt=0, description="Noninteractive training summary interval")
    save_every_updates: int = Field(
        default=100, gt=0, description="Save portable weights and a separate resume snapshot every N updates"
    )
    max_resume_checkpoints: int = Field(default=1, ge=0, description="Keep the newest N resume snapshots; 0 keeps all")
    max_weight_checkpoints: int = Field(default=0, ge=0, description="Keep the newest N weight sets; 0 keeps all")
    tensorboard: str | None = Field(default=None, min_length=1, pattern=r"\S")
    save_precision: FloatingPrecision = Field(
        default="fp32",
        description="Floating precision of published model/adapter/encoder artifacts; resume state retains training dtypes",
    )
    checkpoint_storage: Literal["complete", "frozen-reference"] = "complete"
    _run_name: str | None = PrivateAttr(default=None)
