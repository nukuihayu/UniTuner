"""Apply interleaved FP32 rotary matrices with an eager CUDA fusion boundary."""

from importlib.util import find_spec

import torch
from torch import Tensor

_TRITON_AVAILABLE = find_spec("triton") is not None


def _native_rotary(values: Tensor, positions: Tensor) -> Tensor:
    """Keep each multiplication and addition in FP32 before restoring input dtype."""
    pairs = values.float().reshape(*values.shape[:-1], -1, 1, 2)
    result = positions[..., 0] * pairs[..., 0] + positions[..., 1] * pairs[..., 1]
    return result.reshape_as(values).to(values)


class _Rotary(torch.autograd.Function):
    """Save only the constant matrix; higher derivatives retain native autograd."""

    @staticmethod
    def forward(ctx, values: Tensor, positions: Tensor) -> Tensor:
        from unitunner.models.components.rotary_cuda import rotary_cuda

        ctx.save_for_backward(positions)
        return rotary_cuda(values, positions)

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        (gradient,) = grad_outputs
        (positions,) = ctx.saved_tensors
        if torch.is_grad_enabled():
            return _native_rotary(gradient, positions.transpose(-1, -2)), None
        from unitunner.models.components.rotary_cuda import rotary_cuda

        return rotary_cuda(gradient, positions, transpose=True), None


def apply_rotary(values: Tensor, positions: Tensor) -> Tensor:
    """Rotate BHLD values using Bx1xLx(D/2)x2x2 FP32 matrices.

    Eager CUDA uses one kernel for each forward/backward, with separate FP32
    products and sums (no FMA contraction). CPU, compilation, trainable positions,
    and other broadcasting/layout contracts retain the native tensor expression.
    Triton is loaded lazily and is not required for the native path.
    """
    if (
        not torch.compiler.is_compiling()
        and values.device.type == "cuda"
        and _TRITON_AVAILABLE
        and values.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and positions.dtype == torch.float32
        and values.device == positions.device
        and not positions.requires_grad
        and values.ndim == 4
        and min(values.shape) > 0
        and values.shape[-1] % 2 == 0
        and positions.shape == (values.shape[0], 1, values.shape[2], values.shape[3] // 2, 2, 2)
        and positions.is_contiguous()
        and not torch._C._are_functorch_transforms_active()
        and torch.autograd.forward_ad.unpack_dual(values).tangent is None
        and torch.autograd.forward_ad.unpack_dual(positions).tangent is None
    ):
        return _Rotary.apply(values, positions)
    return _native_rotary(values, positions)
