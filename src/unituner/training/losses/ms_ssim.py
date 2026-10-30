"""Five-scale RGB MS-SSIM supervision without external feature networks."""

import torch
from torch import Tensor
from torch.nn import functional as F


def ms_ssim_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Return 1-MS-SSIM for BCHW RGB in nominal [-1,1], using FP32 statistics.

    Use an 11-pixel Gaussian (sigma 1.5), valid separable filtering, population
    covariance, K1=.01/K2=.03 and the standard five exponents. Odd pyramid
    dimensions use parity padding in 2x average pooling. Both axes must exceed
    160 pixels. Convert to nominal [0,1] without clipping decoder overshoots.
    Clamp scale scores to [1e-6,1] before fractional powers to keep gradients
    finite for anticorrelated images. Targets never receive gradients.
    """
    if (
        prediction.ndim != 4
        or prediction.shape != target.shape
        or prediction.shape[1] != 3
        or not prediction.numel()
        or min(prediction.shape[-2:]) <= 160
        or prediction.device != target.device
        or not prediction.is_floating_point()
        or not target.is_floating_point()
    ):
        raise ValueError("MS-SSIM requires matching floating RGB BCHW images with both dimensions >160")
    with torch.autocast(prediction.device.type, enabled=False):
        x, y = (prediction.float() + 1) * 0.5, (target.detach().float() + 1) * 0.5
        coordinates = torch.arange(11, device=x.device, dtype=torch.float32) - 5
        kernel = torch.exp(-coordinates.square() / (2 * 1.5**2))
        kernel = (kernel / kernel.sum()).reshape(1, 1, 1, 11).expand(3, 1, 1, 11).contiguous()

        def smooth(value: Tensor) -> Tensor:
            return F.conv2d(F.conv2d(value, kernel.transpose(2, 3), groups=3), kernel, groups=3)

        scores = []
        for level in range(5):
            mean_x, mean_y = smooth(x), smooth(y)
            variance_x = smooth(x.square()) - mean_x.square()
            variance_y = smooth(y.square()) - mean_y.square()
            covariance = smooth(x * y) - mean_x * mean_y
            cs = (2 * covariance + 0.03**2) / (variance_x + variance_y + 0.03**2)
            if level == 4:
                luminance = (2 * mean_x * mean_y + 0.01**2) / (mean_x.square() + mean_y.square() + 0.01**2)
                scores.append((luminance * cs).mean((-2, -1)))
            else:
                scores.append(cs.mean((-2, -1)))
                padding = (x.shape[-2] % 2, x.shape[-1] % 2)
                x, y = (F.avg_pool2d(value, 2, padding=padding) for value in (x, y))
        exponents = x.new_tensor((0.0448, 0.2856, 0.3001, 0.2363, 0.1333)).reshape(5, 1, 1)
        similarity = torch.stack(scores).clamp(1e-6, 1).pow(exponents).prod(0).mean()
        return 1 - similarity
