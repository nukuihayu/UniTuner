"""Configure paired Wan2.2 denoisers for a single training run."""

from pydantic import BaseModel, ConfigDict, Field


class WanExpertTrainingConfig(BaseModel):
    """Use model.weights as the low-noise expert and a second high-noise source."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    high_noise_weights: str = Field(min_length=1)
    boundary: float = Field(gt=0, lt=1)
