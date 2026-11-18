"""Fuse Hadamard scaling while retaining every native dtype rounding boundary."""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _product_kernel(
    left, right, scale, gradient, output, grad_right, size: tl.constexpr, backward: tl.constexpr, block: tl.constexpr
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < size
    dtype = output.dtype.element_ty
    a = tl.load(left + index, valid, 0).to(tl.float32)
    b = tl.load(right + index, valid, 0).to(tl.float32)
    # Torch's scalar-tensor promotion converts the scalar to the matrix dtype.
    s = tl.load(scale).to(dtype).to(tl.float32)
    if backward:
        g = tl.load(gradient + index, valid, 0).to(tl.float32)
        scaled = (g * s).to(dtype).to(tl.float32)
        tl.store(output + index, scaled * b, valid)
        tl.store(grad_right + index, scaled * a, valid)
    else:
        product = (a * b).to(dtype).to(tl.float32)
        tl.store(output + index, product * s, valid)


def product_cuda(
    left: Tensor, right: Tensor, scale: Tensor, gradient: Tensor | None = None
) -> tuple[Tensor, Tensor | None]:
    """Evaluate a contiguous product or its two input gradients without temporaries."""
    output = torch.empty_like(left)
    other = None if gradient is None else torch.empty_like(right)
    with torch.cuda.device(left.device):
        launch = cast(Callable[..., None], _product_kernel[(triton.cdiv(left.numel(), 256),)])
        launch(
            left, right, scale, gradient, output, other, left.numel(), gradient is not None, 256, enable_fp_fusion=False
        )
    return output, other
