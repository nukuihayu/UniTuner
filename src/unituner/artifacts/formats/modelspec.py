"""Build minimal ModelSpec strings without inferring model family or owning tensors."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Describe consumer-facing architecture explicitly; preserve caller spelling."""

    architecture: str
    implementation: str
    title: str

    def metadata(self) -> dict[str, str]:
        """Return ModelSpec 1.0 fields, distinct from reload and provenance data."""
        if not self.architecture or not self.implementation or not self.title:
            raise ValueError("ModelSpec architecture, implementation and title must be nonempty")
        return {
            "modelspec.sai_model_spec": "1.0.0",
            "modelspec.architecture": self.architecture,
            "modelspec.implementation": self.implementation,
            "modelspec.title": self.title,
        }
