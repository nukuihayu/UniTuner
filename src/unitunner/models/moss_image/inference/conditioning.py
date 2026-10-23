"""Encode ordered Moss-Image-Edit references using the native packed VAE."""

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor

from unitunner.data.media.image import read_image
from unitunner.models.flux2.vae import Flux2VAE


@torch.no_grad()
def encode_references(vae: Flux2VAE, paths: tuple[str, ...], *, device: torch.device) -> tuple[Tensor, ...]:
    """Keep reference sizes, rounding down to /16; return normalized FP32 CHW means on CPU.

    No semantic-image text encoding or FLUX.2 reference resizing policy applies
    to the text-start Moss-Image-Edit contract. Always park the borrowed VAE on CPU.
    """
    values = []
    try:
        vae.to(device)
        with torch.autocast(device.type, enabled=False):
            for name in paths:
                path = Path(name)
                with Image.open(path) as image:
                    width, height = image.size
                if min(width, height) < 16:
                    raise ValueError("MOSS reference images must be at least 16 pixels on each side")
                pixels = read_image(path, (width // 16 * 16, height // 16 * 16))
                values.append(vae.encode(pixels[None].to(device))[0].float().cpu())
    finally:
        vae.cpu()
    return tuple(values)
