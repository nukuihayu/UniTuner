"""Declare native Wan UMT5 encoder topology independently of tokenizer assets."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WanT5Config(BaseModel):
    """Describe native UMT5-XXL; shared bias is an explicit alternative topology."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    vocab_size: int = Field(default=256384, gt=0)
    hidden_size: int = Field(default=4096, gt=0)
    attention_dim: int = Field(default=4096, gt=0)
    ffn_dim: int = Field(default=10240, gt=0)
    num_heads: int = Field(default=64, gt=0)
    num_layers: int = Field(default=24, gt=0)
    num_buckets: int = Field(default=32, ge=4)
    shared_position_bias: bool = False
    dropout: float = Field(default=0.1, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Require integral attention heads and native even signed bucket halves."""
        if self.attention_dim % self.num_heads or self.num_buckets % 4 or self.num_buckets >= 512:
            raise ValueError("Wan T5 requires integral attention heads and num_buckets divisible by4 below512")
        return self
