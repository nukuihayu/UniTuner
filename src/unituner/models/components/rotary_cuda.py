"""Fuse eager rotary products without changing native FP32 rounding order."""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rotary_kernel(
    values,
    positions,
    output,
    heads: tl.constexpr,
    length: tl.constexpr,
    width: tl.constexpr,
    pairs: tl.constexpr,
    batch_stride: tl.constexpr,
    head_stride: tl.constexpr,
    length_stride: tl.constexpr,
    width_stride: tl.constexpr,
    position_batch_stride: tl.constexpr,
    position_length_stride: tl.constexpr,
    transpose: tl.constexpr,
    block: tl.constexpr,
):
    """Walk BLHD storage, reading arbitrary BHLD input strides without a copy."""
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < pairs
    pair = index % (width // 2)
    head = index // (width // 2) % heads
    token = index // (width // 2 * heads) % length
    batch = index // (width // 2 * heads * length)
    source = batch * batch_stride + head * head_stride + token * length_stride + 2 * pair * width_stride
    matrix = batch * position_batch_stride + token * position_length_stride + 4 * pair
    first = tl.load(values + source, valid, 0).to(tl.float32)
    second = tl.load(values + source + width_stride, valid, 0).to(tl.float32)
    p00 = tl.load(positions + matrix, valid, 0)
    p01 = tl.load(positions + matrix + 1, valid, 0)
    p10 = tl.load(positions + matrix + 2, valid, 0)
    p11 = tl.load(positions + matrix + 3, valid, 0)
    if transpose:
        out_first = first * p00 + second * p10
        out_second = first * p01 + second * p11
    else:
        out_first = first * p00 + second * p01
        out_second = first * p10 + second * p11
    tl.store(output + 2 * index, out_first, valid)
    tl.store(output + 2 * index + 1, out_second, valid)


def rotary_cuda(values: Tensor, positions: Tensor, *, transpose: bool = False) -> Tensor:
    """Return dense BLHD-backed BHLD values, matching native attention's usual layout."""
    batch, heads, length, width = values.shape
    output = torch.empty((batch, length, heads, width), dtype=values.dtype, device=values.device).transpose(1, 2)
    pairs = values.numel() // 2
    with torch.cuda.device(values.device):
        # Triton's launch options extend the JIT function's Python signature.
        launch = cast(Callable[..., None], _rotary_kernel[(triton.cdiv(pairs, 256),)])
        launch(
            values,
            positions,
            output,
            heads,
            length,
            width,
            pairs,
            *values.stride(),
            positions.stride(0),
            positions.stride(2),
            transpose,
            256,
            enable_fp_fusion=False,
        )
    return output
