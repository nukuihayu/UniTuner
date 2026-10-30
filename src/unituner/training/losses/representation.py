"""REPA projection alignment and stopped-gradient LayerSync cosine losses."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class VisualGrid:
    """Positive [T, H, W] target grid for row-major representation alignment.

    Dimensions exclude text, reference-image and padding tokens. This metadata
    describes the loss target; it does not own model topology or recorded tensors.
    """

    frames: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if min(self.frames, self.height, self.width) <= 0:
            raise ValueError("Visual-grid dimensions must be positive")

    @property
    def tokens(self) -> int:
        """Return the number of visual tokens in the flattened grid."""
        return self.frames * self.height * self.width


def layersync_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity over [B, ..., D]; only student receives gradients."""
    if student.shape != teacher.shape or student.ndim < 3:
        raise ValueError("LayerSync requires equal [B, ..., D] feature shapes")
    return -F.cosine_similarity(student.float(), teacher.detach().float(), dim=-1).mean()


class RepaProjection(nn.Module):
    """Training-owned FP32 LayerNorm/linear projection into frozen encoder space."""

    def __init__(self, hidden_dim: int, encoder_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0 or encoder_dim <= 0:
            raise ValueError("REPA projection dimensions must be positive")
        self.encoder_dim = encoder_dim
        self.layers = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, encoder_dim))

    def alignment_loss(self, hidden: torch.Tensor, target: torch.Tensor, grid: VisualGrid) -> torch.Tensor:
        """Align student tokens to a frozen encoder's temporal/spatial feature grid.

        Args:
            hidden: Floating student features [B, S, D], with S equal to the target
                grid's token count and D equal to the projection's input width.
            target: Finite cached encoder features [B, T, H, W, E] from the same
                crop and frames as the latent cache, excluding CLS tokens.
            grid: Positive target [T, H, W] dimensions in row-major token order.

        Returns:
            Scalar FP32 negative mean cosine similarity. Gradients reach the
            student and projector, while the encoder target is detached. Trilinear
            interpolation resizes temporal/spatial axes without mixing channels.

        Raises:
            ValueError: Batch, token count, encoder width or target values violate
                the alignment contract. Projector weights must remain FP32 and
                reside on the student's device; the training owner manages them.
        """
        if hidden.ndim != 3 or hidden.shape[1] != grid.tokens:
            raise ValueError("REPA hidden token count disagrees with its visual grid")
        if target.ndim != 5 or target.shape[0] != hidden.shape[0]:
            raise ValueError("REPA targets must have shape [B, T, H, W, encoder_dim]")
        if target.shape[-1] != self.encoder_dim or not torch.isfinite(target).all():
            raise ValueError("REPA encoder dimension mismatch or non-finite targets")
        # Disable autocast: trainable alignment stays FP32 even with BF16 model forwards.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            projected = self.layers(hidden.float())
            target = target.detach().to(device=hidden.device, dtype=torch.float32).permute(0, 4, 1, 2, 3)
            target = F.interpolate(
                target, size=(grid.frames, grid.height, grid.width), mode="trilinear", align_corners=False
            )
            target = target.permute(0, 2, 3, 4, 1).reshape_as(projected)
            return -F.cosine_similarity(projected, target, dim=-1).mean()
