"""Apply non-reentrant recomputation using explicit invocation regions."""

from collections.abc import Callable
from contextlib import nullcontext

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from unitunner.execution.activations import AttentionReplay
from unitunner.execution.regions import ModelRegion
from unitunner.models.components.forward_override import ForwardOverride


def run_region(region: ModelRegion, inputs: Tensor, *, recompute_interval: int | None = None) -> Tensor:
    """Run a tensor-to-tensor region, preserving RNG and parameter/input gradients.

    Selection uses this invocation's logical index, never copied module policy
    attributes. The caller retains stable weight storage until backward finishes.
    Structured multi-stream regions require their own typed invocation boundary.
    """
    if recompute_interval is not None and recompute_interval < 1:
        raise ValueError("recompute_interval must be positive")
    if (
        recompute_interval is not None
        and region.logical_index % recompute_interval == 0
        and region.module.training
        and torch.is_grad_enabled()
    ):
        return checkpoint(region.module, inputs, use_reentrant=False, preserve_rng_state=True)
    return region.module(inputs)


class RecomputeInstallation[R: Tensor | tuple[Tensor, ...], **P]:
    """Install a typed recomputation boundary before compiling the outer callable.

    The region describes this exact invocation, including logical index and
    branch. Reused modules with different per-call policies should use explicit
    run_region calls instead. The owner keeps the original callable alive until
    backward completes and restores only its own forward mutation.
    """

    def __init__(
        self, region: ModelRegion, *, interval: int = 1, attention_offload: bool = False, require_training: bool = True
    ) -> None:
        if interval < 1:
            raise ValueError("Recompute interval must be positive")
        self.region: ModelRegion = region
        original: Callable[P, R] = region.module.forward

        def installed(*args: P.args, **kwargs: P.kwargs) -> R:
            if (
                region.logical_index % interval == 0
                and (region.module.training or not require_training)
                and torch.is_grad_enabled()
            ):
                # Pass actual tensors so Torch captures CUDA RNG for replay.
                # Its reserved-option stub cannot narrow native ParamSpec kwargs.
                if not attention_offload:
                    return checkpoint(original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)  # ty: ignore[invalid-argument-type]
                return checkpoint(
                    original,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    context_fn=lambda: (nullcontext(), AttentionReplay()),
                    **kwargs,  # ty: ignore[invalid-argument-type]
                )
            return original(*args, **kwargs)

        self._forward = ForwardOverride(region.module, installed, owner="RecomputeInstallation")

    def close(self) -> None:
        """Restore the preceding callable in reverse installation order."""
        self._forward.close()
