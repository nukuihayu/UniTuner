"""Fuse LoRA scaling and addition with an explicit intermediate dtype cast."""

from collections.abc import Callable
from typing import cast

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _scaled_residual_kernel(base, delta, scale, output, size: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    valid = index < size
    dtype = output.dtype.element_ty
    value = tl.load(base + index, valid, 0).to(tl.float32)
    update = tl.load(delta + index, valid, 0).to(tl.float32)
    factor = tl.load(scale).to(dtype).to(tl.float32)
    scaled = (update * factor).to(dtype).to(tl.float32)
    tl.store(output + index, value + scaled, valid)


def scaled_residual_cuda(base: Tensor, delta: Tensor, scale: Tensor) -> Tensor:
    """Evaluate base + round(delta * scale) for matching contiguous CUDA tensors."""
    output = torch.empty_like(base)
    with torch.cuda.device(base.device):
        launch = cast(Callable[..., None], _scaled_residual_kernel[(triton.cdiv(base.numel(), 256),)])
        launch(base, delta, scale, output, base.numel(), 256, enable_fp_fusion=False)
    return output
