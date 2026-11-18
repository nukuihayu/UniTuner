"""DWT band loss adapted from rockerBOO/wavelet-loss (Apache-2.0).

Reference ef5d88cfb30cf40c5a5cdb52ed657ee75449c70a. Modified for typed
configuration, fixed filters, batched separable convolution and no CPU metrics.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unituner.config.wavelet import WaveletLossConfig

_FILTERS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "haar": ((0.7071067811865476, 0.7071067811865476), (-0.7071067811865476, 0.7071067811865476)),
    "db4": (
        (
            -0.010597401785069032,
            0.0328830116668852,
            0.030841381835560764,
            -0.18703481171909309,
            -0.027983769416859854,
            0.6308807679298589,
            0.7148465705529157,
            0.2303778133088965,
        ),
        (
            -0.2303778133088965,
            0.7148465705529157,
            -0.6308807679298589,
            -0.027983769416859854,
            0.18703481171909309,
            0.030841381835560764,
            -0.0328830116668852,
            -0.010597401785069032,
        ),
    ),
}


class WaveletLoss(nn.Module):
    """Sum per-band MSE with zero-padded DWT and optional per-sample sigma fade.

    Inputs use BCHW, or BCFHW with frames treated independently. Arithmetic is
    FP32 (FP64 inputs retain FP64); targets remain differentiable, as upstream.
    LL contributes only after the effective threshold. Band normalization uses
    shared target mean/std. No host readback, target detach or range conversion
    occurs. The caller supplies validated noise fractions, never native times.
    """

    def __init__(self, config: WaveletLossConfig, *, device: torch.device) -> None:
        super().__init__()
        self.config = config
        # Upstream creates filters in default FP32 before converting to the input dtype.
        self.register_buffer(
            "filters", torch.tensor(_FILTERS[config.wavelet], dtype=torch.float32, device=device).flip(1)
        )

    def _axis(self, value: Tensor, kernel: Tensor, axis: int) -> Tensor:
        """Keep upstream convolution shape/order and decimation storage on CUDA."""
        size = kernel.numel()
        padding = (0, 0, size - 1, size - 1) if axis == 2 else (size - 1, size - 1, 0, 0)
        weight = kernel.view(1, 1, -1, 1) if axis == 2 else kernel.view(1, 1, 1, -1)
        filtered = F.conv2d(F.pad(value, padding), weight)
        return filtered[:, :, 1::2, :].contiguous() if axis == 2 else filtered[:, :, :, 1::2].contiguous()

    def _decompose(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply full convolution then decimate at 1::2, preserving pywt band order."""
        batch, channels, height, width = value.shape
        filters = self.get_buffer("filters").to(value.dtype)
        source = value.reshape(batch * channels, 1, height, width)
        low, high = self._axis(source, filters[0], 2), self._axis(source, filters[1], 2)
        bands = (
            self._axis(low, filters[0], 3),
            self._axis(high, filters[0], 3),
            self._axis(low, filters[1], 3),
            self._axis(high, filters[1], 3),
        )
        ll, lh, hl, hh = (band.reshape(batch, channels, *band.shape[2:]) for band in bands)
        return ll, lh, hl, hh

    def _compute(self, prediction: Tensor, target: Tensor, sigma: Tensor | None = None) -> Tensor:
        """Return the weighted scalar band sum, preserving per-frame noise weighting."""
        if (
            prediction.shape != target.shape
            or prediction.device != target.device
            or prediction.ndim not in (4, 5)
            or not prediction.numel()
        ):
            raise ValueError("Wavelet loss requires matching nonempty BCHW/BCFHW tensors on one device")
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise ValueError("Wavelet loss requires floating prediction and target")
        batch = prediction.shape[0]
        if sigma is not None and (sigma.shape != (batch,) or sigma.device != prediction.device):
            raise ValueError("Wavelet sigma must be [B] on the prediction device")
        frames = 1
        if prediction.ndim == 5:
            frames = prediction.shape[2]
            prediction = prediction.transpose(1, 2).flatten(0, 1)
            target = target.transpose(1, 2).flatten(0, 1)
        dtype = torch.float64 if torch.float64 in (prediction.dtype, target.dtype) else torch.float32
        prediction, target = prediction.to(dtype), target.to(dtype)
        fade = torch.ones(batch, device=prediction.device)
        if sigma is not None:
            fade *= torch.sigmoid((self.config.timestep_cutoff - sigma) * 4.0 / self.config.timestep_transition_width)
        fade = fade.repeat_interleave(frames)
        threshold = self.config.ll_level_threshold
        if threshold is not None and threshold <= 0:
            threshold += self.config.levels
        total = prediction.new_zeros(())
        for level in range(self.config.levels):
            pred_bands = self._decompose(prediction)
            target_bands = self._decompose(target)
            for index, weight in enumerate((self.config.ll, self.config.lh, self.config.hl, self.config.hh)):
                if not weight or (index == 0 and threshold is not None and level + 1 <= threshold):
                    continue
                pred, expected = pred_bands[index], target_bands[index]
                if self.config.normalize_bands:
                    mean = expected.mean()
                    std = (expected.std() if expected.numel() > 1 else expected.new_zeros(())) + 1e-8
                    pred, expected = (pred - mean) / std, (expected - mean) / std
                total = total + ((fade * weight)[:, None, None, None] * (pred - expected).square()).mean()
            prediction, target = pred_bands[0], target_bands[0]
        return total * self.config.weight

    def compute(self, prediction: Tensor, target: Tensor, sigma: Tensor | None = None) -> Tensor:
        """Evaluate the objective outside model autocast to preserve FP32 band arithmetic."""
        with torch.autocast(prediction.device.type, enabled=False):
            return self._compute(prediction, target, sigma)

    forward = compute


def image_flow_loss(prediction: Tensor, target: Tensor, sigma: Tensor, wavelet: WaveletLoss | None) -> Tensor:
    """Preserve native element-mean velocity MSE and optionally add DWT supervision."""
    loss = F.mse_loss(prediction.float(), target.float())
    return loss if wavelet is None else loss + wavelet(prediction, target, sigma)
