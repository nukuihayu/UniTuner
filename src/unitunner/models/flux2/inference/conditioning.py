"""Encode native FLUX.2 reference images with caller-owned packed VAE resources."""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from unitunner.models.flux2.data.media import read_reference_image
from unitunner.models.flux2.vae import Flux2VAE


@torch.no_grad()
def build_flux2_references(vae: Flux2VAE, paths: Sequence[Path], *, original: bool = False) -> tuple[Tensor, ...]:
    """Return ordered normalized CHW means rounded BF16 then stored as CPU FP32.

    Preserve native single2024²/multiple1024² aspect-bucket preprocessing and
    original-size override. The packed VAE uses its frozen BN statistics;
    posterior sampling is not performed. Caller owns VAE placement/modes, and
    global RNG is untouched. Reference dimensions are independent of target size.
    """
    parameter = next(vae.parameters())
    with torch.autocast(parameter.device.type, enabled=False):
        return tuple(
            vae.encode(read_reference_image(path, count=len(paths), original=original)[None].to(parameter.device))[0]
            .to(torch.bfloat16)
            .float()
            .cpu()
            for path in paths
        )
