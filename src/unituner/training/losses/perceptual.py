"""Frozen native DINOv3 perception with generated-image gradients."""

import torch
from torch import nn

from unituner.models.features.convnext import DinoConvNeXt
from unituner.models.features.normalization import normalize_image
from unituner.training.losses.dists import dists_loss


class DinoPerceptualLoss(nn.Module):
    """DINOv3 channel-uniform DISTS over RGB and three spatial feature levels.

    The caller owns the shared frozen ConvNeXt backbone, device and FP32 dtype.
    Target features are evaluated without autograd. Generated-image features
    retain their graph through the frozen backbone; BF16 AMP is optional.
    """

    def __init__(self, backbone: DinoConvNeXt, *, bf16_compute: bool = False) -> None:
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.bf16_compute = bf16_compute

    def compute(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar FP32 perception loss for matching RGB BCHW [-1, 1] pairs."""
        with torch.autocast(
            generated.device.type, dtype=torch.bfloat16, enabled=self.bf16_compute, cache_enabled=False
        ):
            left = ((generated.float() + 1) / 2, *self.backbone.features(normalize_image(generated), pooled=True))
            with torch.no_grad():
                right = (
                    (target.detach().float() + 1) / 2,
                    *self.backbone.features(normalize_image(target.detach()), pooled=True),
                )
            return dists_loss(left, right)

    forward = compute
