"""Describe one loss phase without coupling methods to parameter selection."""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass(frozen=True, slots=True)
class StepResult:
    """Carry a scalar mean loss and actual sample count for one update phase.

    The caller retains the graph only until backward. Unequal microbatch sizes
    are normalized by sample count. GAN phases call the engine separately for
    their respective optimizer; no multi-phase restoration method is implied.
    """

    loss: Tensor
    sample_count: int
    phase: str = "main"
    metrics: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.loss.ndim != 0 or not self.loss.requires_grad or self.sample_count <= 0 or not self.phase:
            raise ValueError("StepResult requires a differentiable scalar mean, samples and a phase")
