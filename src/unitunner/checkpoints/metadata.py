"""Serialize training provenance for published artifacts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from unitunner.artifacts.formats.precision import FloatingPrecision


class Provenance(BaseModel):
    """Record redacted summaries separately from loss/update and reload state."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal[1] = 1
    model_family: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    method: str = Field(min_length=1)
    parameter_mode: Literal["full", "adapter"]
    precision: FloatingPrecision
    completed_updates: int = Field(ge=0)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def metadata(self) -> dict[str, str]:
        """Return string-only safetensors metadata without model reload fields."""
        return {"unitunner.provenance": self.model_dump_json()}
