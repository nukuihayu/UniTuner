"""Control cache execution independently of training microbatches."""

from pydantic import BaseModel, ConfigDict, Field


class CacheOptions(BaseModel):
    """Bound same-resolution encoder batches; one retains the original execution path."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    batch_size: int = Field(default=1, gt=0)
