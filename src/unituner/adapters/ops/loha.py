"""Apply scaled Hadamard products with exact eager low-precision intermediates."""

from importlib.util import find_spec

import torch
from torch import Tensor

_TRITON_AVAILABLE = find_spec("triton") is not None


class _ScaledProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left: Tensor, right: Tensor, scale: Tensor) -> Tensor:
        from unituner.adapters.ops.loha_cuda import product_cuda

        ctx.save_for_backward(left, right, scale)
        return product_cuda(left, right, scale)[0]

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor, Tensor, None]:
        (gradient,) = grad_outputs
        left, right, scale = ctx.saved_tensors
        if torch.is_grad_enabled():
            scaled = gradient * scale
            return scaled * right, scaled * left, None
        from unituner.adapters.ops.loha_cuda import product_cuda

        grad_left, grad_right = product_cuda(left, right, scale, gradient.contiguous())
        assert grad_right is not None
        return grad_left, grad_right, None


def scaled_product(left: Tensor, right: Tensor, scale: Tensor) -> Tensor:
    """Fuse (left * right) * scale on CUDA; retain native transforms and scale gradients."""
    if (
        not torch.compiler.is_compiling()
        and _TRITON_AVAILABLE
        and left.device.type == "cuda"
        and left.device == right.device == scale.device
        and left.dtype == right.dtype
        and left.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and scale.dtype in (left.dtype, torch.float32)
        and left.shape == right.shape
        and left.ndim > 0
        and left.numel() > 0
        and left.is_contiguous()
        and right.is_contiguous()
        and scale.ndim == 0
        and not scale.requires_grad
        and not torch._C._are_functorch_transforms_active()
        and all(torch.autograd.forward_ad.unpack_dual(t).tangent is None for t in (left, right, scale))
    ):
        return _ScaledProduct.apply(left, right, scale)
    return (left * right) * scale
