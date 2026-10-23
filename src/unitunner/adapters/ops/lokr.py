"""Preserve dense LoKr arithmetic while fusing Kronecker construction and casts."""

from importlib.util import find_spec

import torch
from torch import Tensor

_TRITON_AVAILABLE = find_spec("triton") is not None


class _Kronecker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left: Tensor, right: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
        from unitunner.adapters.ops.lokr_cuda import kronecker_cuda

        ctx.save_for_backward(left, right, scale)
        return kronecker_cuda(left, right, scale, dtype)

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor, Tensor, None, None]:
        (gradient,) = grad_outputs
        left, right, scale = ctx.saved_tensors
        # Match MulBackward followed by the two broadcast reductions in kron.
        # Native strided products choose a different reduction layout; retain
        # that path, along with the differentiable higher-order expression.
        if not torch.is_grad_enabled() and gradient.is_contiguous():
            from unitunner.adapters.ops.lokr_cuda import kronecker_products_cuda

            product_left, product_right = kronecker_products_cuda(gradient, left, right, scale)
            grad_left = product_left.sum((1, 3))
            del product_left
            grad_right = product_right.sum((0, 2)).to(right.dtype)
            return grad_left, grad_right, None, None
        expanded = (gradient.float() * scale).reshape(left.shape[0], right.shape[0], left.shape[1], right.shape[1])
        grad_left = (expanded * right[None, :, None, :]).sum((1, 3))
        grad_right = (expanded * left[:, None, :, None]).sum((0, 2)).to(right.dtype)
        return grad_left, grad_right, None, None


def scaled_kronecker(left: Tensor, right: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    """Fuse dense construction for FP32 masters without factoring the linear contraction."""
    if (
        not torch.compiler.is_compiling()
        and _TRITON_AVAILABLE
        and left.device.type == "cuda"
        and left.device == right.device == scale.device
        and left.dtype == scale.dtype == torch.float32
        and right.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and dtype in (torch.float32, torch.float16, torch.bfloat16)
        and left.ndim == right.ndim == 2
        and left.numel() > 0
        and right.numel() > 0
        and left.is_contiguous()
        and right.is_contiguous()
        and scale.ndim == 0
        and not scale.requires_grad
        and not torch._C._are_functorch_transforms_active()
        and all(torch.autograd.forward_ad.unpack_dual(t).tangent is None for t in (left, right, scale))
    ):
        return _Kronecker.apply(left, right, scale, dtype)
    return (torch.kron(left.contiguous(), right.contiguous()) * scale).to(dtype)
