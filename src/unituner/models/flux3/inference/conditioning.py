"""Encode official FLUX3 reference images with explicit caller-owned packed VAE resources."""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from unituner.models.flux2.vae import Flux2VAE
from unituner.models.flux3.data.media import read_reference_images


@torch.no_grad()
def build_flux3_references(vae: Flux2VAE, paths: Sequence[Path]) -> tuple[Tensor, ...]:
    """Return CHW packed-VAE means rounded to BF16 then stored as CPU FP32.

    Apply the official single/multi-reference preprocessing and frozen VAE
    normalization, then preserve official reference feature rounding. No posterior
    sampling or RNG consumption occurs. Caller owns VAE residency and modes;
    reference shapes remain independent from the target generation grid.
    """
    parameter = next(vae.parameters())
    with torch.autocast(parameter.device.type, enabled=False):
        return tuple(
            vae.encode(reference.pixels.unsqueeze(0).to(parameter.device))[0].to(torch.bfloat16).float().cpu()
            for reference in read_reference_images(paths)
        )
