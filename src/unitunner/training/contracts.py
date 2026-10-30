"""Define shared image-flow data, update results and training counters."""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass(slots=True)
class RunState:
    """Count consumed microbatches, samples and successful optimizer updates."""

    updates: int = 0
    microbatches: int = 0
    samples: int = 0


@dataclass(frozen=True, slots=True)
class ImageFlowRequest:
    """Carry normalized BCHW latents, B noise fractions and BLD captions."""

    latents: Tensor
    noise: Tensor
    sigma: Tensor
    captions: Tensor
    caption_mask: Tensor | None = None
    references: tuple[Tensor, ...] = ()


@dataclass(frozen=True, slots=True)
class ImageFlowPrediction:
    """Carry native velocity, target and timesteps without detaching."""

    prediction: Tensor
    target: Tensor
    timesteps: Tensor


@dataclass(frozen=True, slots=True)
class StepResult:
    """Carry a scalar mean loss and actual sample count for one update phase.

    The caller retains the graph only until backward. Unequal microbatch sizes
    are normalized by sample count. Named phases select disjoint optimizers
    within the same completed data update.
    """

    loss: Tensor
    sample_count: int
    phase: str = "main"
    metrics: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.loss.ndim != 0 or not self.loss.requires_grad or self.sample_count <= 0 or not self.phase:
            raise ValueError("StepResult requires a differentiable scalar mean, samples and a phase")
