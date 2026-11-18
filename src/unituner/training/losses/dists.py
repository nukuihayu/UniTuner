"""Channel-uniform deep image structure and texture similarity."""

import torch


def dists_loss(predicted: tuple[torch.Tensor, ...], target: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Channel-uniform DISTS with centered FP32 covariance (equation 11).

    Centering avoids cancellation on almost constant features. Unlike learned
    DISTS variants, every feature channel has equal weight (arXiv:2508.08227v2).

    Args:
        predicted: Nonempty BCHW feature levels on one device. Spatial grids
            and channel counts may differ between levels.
        target: Matching feature shapes; detached internally even if trainable.

    Returns:
        Scalar FP32 loss on the feature device, with gradients only to predicted.

    Raises:
        ValueError: Levels are empty, counts differ or paired shapes are not BCHW.
    """
    if not predicted or len(predicted) != len(target):
        raise ValueError("DISTS requires matching nonempty feature levels")
    similarity = predicted[0].new_zeros((), dtype=torch.float32)
    channels = 0
    for left, right in zip(predicted, target, strict=True):
        if (
            left.shape != right.shape
            or left.ndim != 4
            or not left.numel()
            or left.device != right.device
            or left.device != predicted[0].device
            or not left.is_floating_point()
            or not right.is_floating_point()
        ):
            raise ValueError("DISTS feature pairs must have equal BCHW shapes")
        left, right = left.float(), right.detach().float()
        mx, my = left.mean((-2, -1), keepdim=True), right.mean((-2, -1), keepdim=True)
        dx, dy = left - mx, right - my
        structure = (2 * mx * my + 1e-6) / (mx.square() + my.square() + 1e-6)
        texture = (2 * (dx * dy).mean((-2, -1), keepdim=True) + 1e-6) / (
            dx.square().mean((-2, -1), keepdim=True) + dy.square().mean((-2, -1), keepdim=True) + 1e-6
        )
        similarity = similarity + (structure + texture).sum(1).mean()
        channels += left.shape[1]
    return 1 - similarity / (2 * channels)
