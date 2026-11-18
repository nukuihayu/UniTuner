"""Fuse LoRA residuals without changing GEMMs or low-precision scaling."""

from importlib.util import find_spec

import torch
from torch import Tensor
from torch.nn import functional as F

_TRITON_AVAILABLE = find_spec("triton") is not None


class _ScaledResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base: Tensor, delta: Tensor, scale: Tensor) -> Tensor:
        from unituner.adapters.ops.lora_cuda import scaled_residual_cuda

        ctx.save_for_backward(scale)
        return scaled_residual_cuda(base, delta, scale)

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor, Tensor, None]:
        (gradient,) = grad_outputs
        (scale,) = ctx.saved_tensors
        return gradient, gradient * scale, None


def _scaled_residual(base: Tensor, delta: Tensor, scale: Tensor) -> Tensor:
    if (
        not torch.compiler.is_compiling()
        and _TRITON_AVAILABLE
        and base.device.type == "cuda"
        and base.device == delta.device == scale.device
        and base.dtype == delta.dtype
        and base.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and scale.dtype in (base.dtype, torch.float32)
        and base.shape == delta.shape
        and base.numel() >= 4096
        and base.is_contiguous()
        and delta.is_contiguous()
        and scale.ndim == 0
        and not scale.requires_grad
        and not torch._C._are_functorch_transforms_active()
        and all(torch.autograd.forward_ad.unpack_dual(t).tangent is None for t in (base, delta, scale))
    ):
        return _ScaledResidual.apply(base, delta, scale)
    return base + delta * scale


def fused_lora_linear(
    inputs: Tensor, weight: Tensor, bias: Tensor | None, down: Tensor, up: Tensor, scale: Tensor
) -> Tensor:
    """Evaluate frozen linear plus LoRA with native autograd and live scaling.

    Inputs have shape [..., in_features]; factors have shapes [rank, in_features]
    and [out_features, rank]. Preserve GEMM rounding and graph accumulation order.
    Eligible eager CUDA residuals fuse scaling and addition, retaining the
    intermediate dtype rounding. Compilation retains the native expression.
    """
    base = F.linear(inputs, weight, bias)
    delta = F.linear(F.linear(inputs, down), up)
    return _scaled_residual(base, delta, scale)
