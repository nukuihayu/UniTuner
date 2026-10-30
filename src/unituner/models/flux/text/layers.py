"""Preserve Transformers T5 feed-forward ordering with logical weight dtypes."""

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn

from unituner.models.components.quantized import linear_dtype


class T5TextFeedForward(nn.Module):
    """Keep the original module keys while avoiding activation casts to packed codes.

    Transformers inspects ``wo.weight.dtype`` before the output projection. A
    packed FP8/uint8 weight is storage, so that check must use its floating
    computation dtype. No activation rounding or quantization is introduced.
    """

    wi: nn.Linear
    wi_0: nn.Linear
    wi_1: nn.Linear
    wo: nn.Linear
    dropout: nn.Module
    act: Callable[[Tensor], Tensor]

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        for name, module in source.named_children():
            self.add_module(name, module)
        self.act = cast(Callable[[Tensor], Tensor], source.act)
        self.gated = hasattr(source, "wi_0")
        self.train(source.training)

    def project(self, values: Tensor) -> Tensor:
        if self.gated:
            values = self.act(self.wi_0(values)) * self.wi_1(values)
        else:
            values = self.act(self.wi(values))
        values = self.dropout(values)
        dtype = linear_dtype(self.wo)
        if values.dtype != dtype and dtype != torch.int8:
            values = values.to(dtype)
        return self.wo(values)

    forward = project
