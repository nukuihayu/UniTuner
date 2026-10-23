"""Resolve publication and preview cadence from completed effective updates."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unitunner.config.training import TrainingConfig


@dataclass(frozen=True, slots=True)
class UpdateSchedule:
    """Share final-update and resume rules without owning training or callbacks."""

    max_updates: int
    save_every_updates: int
    preview_every_updates: int | None
    preview_at_start: bool

    @classmethod
    def from_config(cls, config: "TrainingConfig") -> "UpdateSchedule":
        """Read the validated cadence once at the training boundary."""
        preview = config.preview
        return cls(
            config.train.max_updates,
            config.output.save_every_updates,
            None if preview is None else preview.every_updates,
            preview is not None and preview.at_start,
        )

    def initial_preview_due(self, completed_updates: int) -> bool:
        """Render at zero only; resuming must not replay the initial preview."""
        return completed_updates == 0 and self.preview_at_start

    def save_due(self, completed_updates: int) -> bool:
        """Save at the configured cadence and at an off-cadence final update."""
        return self._due(completed_updates, self.save_every_updates)

    def preview_due(self, completed_updates: int) -> bool:
        """Preview after publication, including the final successful update."""
        return self.preview_every_updates is not None and self._due(completed_updates, self.preview_every_updates)

    def _due(self, completed_updates: int, interval: int) -> bool:
        return 0 < completed_updates <= self.max_updates and (
            completed_updates % interval == 0 or completed_updates == self.max_updates
        )
