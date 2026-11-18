"""Evaluate multilevel GAN objectives while isolating discriminator parameters."""

from collections.abc import Generator
from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@contextmanager
def frozen_discriminator(discriminator: nn.Module) -> Generator[None]:
    """Restore parameter gradient flags after a generator forward, including failure.

    Leave modes and spectral-normalization buffers active. Image gradients remain
    enabled; existing discriminator gradients are neither cleared nor modified.
    The caller owns the borrowed module and must finish this forward before D.
    """
    parameters = tuple((parameter, parameter.requires_grad) for parameter in discriminator.parameters())
    discriminator.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)


def gan_loss(logits: tuple[Tensor, ...], real: bool) -> Tensor:
    """Sum per-level FP32 mean BCE logits losses with real label 0.8 and fake 0.

    Levels may have different spatial dimensions; each contributes equally before
    summation. Gradients reach every logits tensor. No target graph is constructed.
    """
    if not logits or any(not value.numel() or not value.is_floating_point() for value in logits):
        raise ValueError("GAN requires nonempty floating logits at every level")
    return torch.stack(
        [
            F.binary_cross_entropy_with_logits(value.float(), torch.full_like(value.float(), 0.8 if real else 0.0))
            for value in logits
        ]
    ).sum()
