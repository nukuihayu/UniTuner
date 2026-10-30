"""Frequency Distribution Loss (Ni et al., CVPR 2024; github.com/eezkni/FDL).

VGG19 stages, 256 normalized 5x5 projections and equal amplitude/phase weights.
Sorting covers every spatial projection. Unlike the reference's unnormalized
FFT, use orthonormal FFTs to avoid resolution-driven amplitude inflation.
"""

import torch
from torch import nn
from torch.autograd.function import FunctionCtx
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class _AngleContext(FunctionCtx):
    saved_tensors: tuple[torch.Tensor, ...]


class _SortedContext(_AngleContext):
    elements: int
    input_dtype: torch.dtype


class _SortedL1(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _SortedContext, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        values_left, indices_left = left.sort(dim=-1)
        values_right = right.sort(dim=-1).values
        difference = values_left - values_right
        ctx.save_for_backward(difference.sign().to(torch.int8), indices_left.to(torch.int32))
        ctx.elements = left.shape[1] * left.shape[2]
        ctx.input_dtype = left.dtype
        return difference.abs().mean((1, 2))

    @staticmethod
    def backward(ctx: _SortedContext, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (gradient,) = grad_outputs
        sign, indices = ctx.saved_tensors
        result = torch.empty_like(sign, dtype=ctx.input_dtype)
        scale = gradient[:, None, None] / ctx.elements
        for start in range(0, sign.shape[1], 8):
            rows = slice(start, start + 8)
            result[:, rows].scatter_(-1, indices[:, rows].long(), (scale * sign[:, rows]).to(ctx.input_dtype))
        return result, None


class _StableAngle(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _AngleContext, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return torch.angle(value)

    @staticmethod
    def backward(ctx: _AngleContext, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (gradient,) = grad_outputs
        (value,) = ctx.saved_tensors
        scale = torch.maximum(value.real.abs(), value.imag.abs())
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        real, imag = value.real / scale, value.imag / scale
        denominator = real.square() + imag.square()
        factor = gradient / torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        return (torch.complex(-imag * factor / scale, real * factor / scale),)


class FrequencyDistributionLoss(nn.Module):
    """Frozen VGG stages and seeded normalized 256xCx5x5 projection banks.

    Input images are RGB BCHW FP32 [-1,1]. Target features run without autograd;
    generated-image gradients traverse the frozen extractor. Projection RNG is
    local and its frozen projection banks are deterministic checkpoint-independent state.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self, stages: nn.ModuleList, channels: tuple[int, ...], seed: int, *, bf16_compute: bool = False
    ) -> None:
        super().__init__()
        if not channels or len(stages) != len(channels) or min(channels) <= 0:
            raise ValueError("Feature stages and channel widths must match")
        self.stages = stages.eval().requires_grad_(False)
        self.bf16_compute = bf16_compute
        self.projections = nn.ParameterList()
        generator = torch.Generator().manual_seed(seed)
        for width in channels:
            projection = torch.randn(256, width, 5, 5, generator=generator)
            projection /= projection.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None, None]
            self.projections.append(nn.Parameter(projection, requires_grad=False))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def features(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Return VGG features with FP32 normalization and optional BF16 convolution."""
        hidden = ((image.float() + 1) * 0.5 - self.mean) / self.std
        outputs = []
        for stage in self.stages:
            hidden = checkpoint(stage, hidden, use_reentrant=False) if hidden.requires_grad else stage(hidden)
            outputs.append(hidden)
        return outputs

    def _projected_distance(self, left: torch.Tensor, right: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        with torch.autocast(left.device.type, dtype=torch.bfloat16, enabled=self.bf16_compute, cache_enabled=False):
            a = F.conv2d(left, projection).float().flatten(2)
            b = F.conv2d(right, projection).float().flatten(2)
        return _SortedL1.apply(a, b)

    def forward_features(self, left: list[torch.Tensor], right: list[torch.Tensor]) -> torch.Tensor:
        """Evaluate the full-spatial distance with orthonormal frequency amplitudes.

        Feature pairs are same-shaped floating BCHW tensors, one per extractor stage.
        The target is detached. Returns a scalar, differentiable to left only.
        """
        losses = []
        for x, y, projection in zip(left, right, self.projections, strict=True):
            if x.ndim != 4 or x.shape != y.shape or not x.numel() or x.device != y.device:
                raise ValueError("FDL requires matching nonempty BCHW feature pairs on one device")
            if not x.is_floating_point() or not y.is_floating_point():
                raise ValueError("FDL features must use floating point")
            x, y = x.float(), y.detach().float()
            # Normalize spectral energy for each feature-map size. This changes
            # amplitude scaling from the reference FDL while preserving phase.
            fft_x = torch.fft.fftn(x, dim=(-2, -1), norm="ortho")
            fft_y = torch.fft.fftn(y, dim=(-2, -1), norm="ortho")
            frequency = x.new_zeros(x.shape[0])
            for a, b, weight in (
                (fft_x.abs(), fft_y.abs(), 1.0),
                (_StableAngle.apply(fft_x), _StableAngle.apply(fft_y), 1.0),
            ):
                if min(a.shape[-2:]) < 5:
                    raise ValueError("FDL requires each feature map to be at least 5x5; VGG19 images must be >=80x80")
                # Bound convolution/sort workspaces in both passes. Recompute each
                # chunk in backward instead of retaining 256 full spatial maps of
                # sort indices and signs for every feature level and FFT branch.
                for bank in projection.split(8):
                    distance = checkpoint(
                        self._projected_distance, a, b, bank, use_reentrant=False, preserve_rng_state=False
                    )
                    frequency = frequency + weight * (bank.shape[0] / projection.shape[0]) * distance
            losses.append(frequency)
        return torch.stack(losses).sum(0).mean()

    def compute(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return FP32 scalar loss for matching BCHW image pairs on the model device."""
        with torch.autocast(
            generated.device.type, dtype=torch.bfloat16, enabled=self.bf16_compute, cache_enabled=False
        ):
            left = self.features(generated)
            with torch.no_grad():
                right = self.features(target.detach())
        with torch.autocast(generated.device.type, enabled=False):
            return self.forward_features(left, right)

    forward = compute
