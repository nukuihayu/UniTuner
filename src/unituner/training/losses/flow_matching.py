"""Batch-local, mixed-precision-safe flow-matching regression losses."""

import torch
from torch import Tensor
from torch.nn import functional as F


def flow_matching_loss(prediction: Tensor, target: Tensor, weights: Tensor | None = None) -> Tensor:
    """Average weighted per-sample velocity errors without cross-sample broadcasting.

    Args:
        prediction: Floating model-native velocities with shape [B, ...].
        target: Matching layout, shape and device; the family owns velocity sign.
        weights: Optional [B] loss weights on the prediction device. Weighting is
            applied to each sample's mean error, then averaged over the batch;
            weights are not normalized by their sum.

    Returns:
        Scalar FP32 loss with gradients to floating inputs and weights. FP64
        inputs retain FP64 arithmetic for high-precision numerical verification.
        FP16/BF16 errors and reductions are evaluated in FP32 to avoid overflow.

    Raises:
        ValueError: Shapes/devices disagree, batch is empty or weights are not [B].
    """
    if prediction.ndim < 2 or prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("Flow matching requires matching nonempty batch-first predictions and targets")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("Flow prediction and target must be floating tensors")
    if prediction.device != target.device:
        raise ValueError("Flow predictions and targets must share a device")
    dtype = torch.float64 if torch.float64 in (prediction.dtype, target.dtype) else torch.float32
    if weights is None:
        # Equal-sized samples: a native mean avoids materializing squared errors.
        return F.mse_loss(prediction.to(dtype), target.to(dtype))
    per_sample = (prediction.to(dtype) - target.to(dtype)).square().flatten(1).mean(1)
    if weights.shape != per_sample.shape or weights.device != prediction.device:
        raise ValueError("Flow weights must be [B] on the prediction device")
    per_sample = per_sample * weights.to(dtype)
    return per_sample.mean()
