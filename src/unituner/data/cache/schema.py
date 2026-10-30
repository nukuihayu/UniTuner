"""Share versioned content identities between cache producers and consumers."""

from pydantic import BaseModel, ConfigDict, Field

from unituner.artifacts.hashing import canonical_sha256


class CacheIdentity(BaseModel):
    """Avoid basename collisions by pinning source, encoder and transform contents.

    Callers hash file contents and all mathematical transform choices, including
    crop, normalization and sampling. No mtime or module path substitutes a hash.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: int = Field(default=1, ge=1, le=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transform_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    layout: str = Field(min_length=1)

    def digest(self) -> str:
        """Return the versioned identity's canonical content address."""
        return canonical_sha256(self.model_dump(mode="json"))
