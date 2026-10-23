"""Track successful optimizer updates independently of artifact metadata."""

from dataclasses import dataclass


@dataclass(slots=True)
class RunState:
    """Track the current phase's consumed microbatches, samples and effective updates."""

    updates: int = 0
    microbatches: int = 0
    samples: int = 0
