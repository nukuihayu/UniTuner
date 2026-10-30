"""Materialize scaled Kronecker weights directly in the linear compute dtype."""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _kronecker_kernel(
    left,
    right,
    scale,
    output,
    size: tl.constexpr,
    in_left: tl.constexpr,
    out_right: tl.constexpr,
    in_right: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < size
    row = index // (in_left * in_right)
    col = index % (in_left * in_right)
    a = tl.load(left + row // out_right * in_left + col // in_right, valid, 0).to(tl.float32)
    b = tl.load(right + row % out_right * in_right + col % in_right, valid, 0).to(tl.float32)
    s = tl.load(scale)
    product = a * b
    tl.store(output + index, product * s, valid)


def kronecker_cuda(left: Tensor, right: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    """Retain FP32 product and scaling, casting only at the linear boundary."""
    shape = (left.shape[0] * right.shape[0], left.shape[1] * right.shape[1])
    output = torch.empty(shape, device=left.device, dtype=dtype)
    with torch.cuda.device(left.device):
        launch = cast(Callable[..., None], _kronecker_kernel[(triton.cdiv(output.numel(), 256),)])
        launch(left, right, scale, output, output.numel(), left.shape[1], *right.shape, 256, enable_fp_fusion=False)
    return output


@triton.jit
def _kronecker_products_kernel(
    gradient,
    left,
    right,
    scale,
    product_left,
    product_right,
    size: tl.constexpr,
    in_left: tl.constexpr,
    out_right: tl.constexpr,
    in_right: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < size
    row = index // (in_left * in_right)
    col = index % (in_left * in_right)
    g = tl.load(gradient + index, valid, 0).to(tl.float32)
    a = tl.load(left + row // out_right * in_left + col // in_right, valid, 0).to(tl.float32)
    b = tl.load(right + row % out_right * in_right + col % in_right, valid, 0).to(tl.float32)
    scaled = g * tl.load(scale)
    tl.store(product_left + index, scaled * b, valid)
    tl.store(product_right + index, scaled * a, valid)


def kronecker_products_cuda(gradient: Tensor, left: Tensor, right: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    """Fuse contiguous gradient products while preserving native FP32 reductions."""
    shape = (left.shape[0], right.shape[0], left.shape[1], right.shape[1])
    product_left = torch.empty(shape, device=gradient.device, dtype=torch.float32)
    product_right = torch.empty_like(product_left)
    with torch.cuda.device(gradient.device):
        launch = cast(Callable[..., None], _kronecker_products_kernel[(triton.cdiv(product_left.numel(), 256),)])
        launch(
            gradient,
            left,
            right,
            scale,
            product_left,
            product_right,
            product_left.numel(),
            left.shape[1],
            *right.shape,
            256,
            enable_fp_fusion=False,
        )
    return product_left, product_right
