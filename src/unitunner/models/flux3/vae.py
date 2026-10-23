# Derived from Black Forest Labs FLUX3 Image Klein, Apache-2.0.
"""Present the shared packed VAE as native FLUX3 image tokens and four-axis IDs."""

from dataclasses import dataclass

import torch
from torch import Tensor

from unitunner.models.flux2.vae import Flux2VAE


@dataclass(frozen=True, slots=True)
class Flux3ImageLatents:
    """Carry one normalized LD image and L4 (zero,row,column,zero) positions.

    Height/width count latent tokens, not pixels. IDs use row-major order and
    remain independent from text/reference time coordinates built by conditioning.
    Tokens retain gradients; no hidden sampling, copy or device transfer occurs.
    """

    tokens: Tensor
    ids: Tensor
    height: int
    width: int


def image_ids(height: int, width: int, *, device: torch.device) -> Tensor:
    """Build native int64 L4 row-major positions on the caller-selected device."""
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("FLUX3 image token dimensions must be positive integers")
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    zeros = torch.zeros_like(rows)
    return torch.stack((zeros, rows, columns, zeros), dim=-1).reshape(height * width, 4)


def encode_image(vae: Flux2VAE, pixels: Tensor) -> Flux3ImageLatents:
    """Encode native CHW RGB [-1,1] to LD means and L4 IDs without sampling.

    FLUX3 and FLUX.2 share the convolution/quant-convolution and packed-BN
    checkpoint topology. The VAE's explicit configuration determines dimensions;
    official FLUX3 uses 32 raw channels, 128 packed channels and a 16-pixel factor.
    This function only presents that shared computation in the official per-image
    token layout. Unlike the official inference-only wrapper, gradients remain
    available for restoration. VAE mode, BN statistics and RNG remain intact.
    """
    if pixels.ndim != 3:
        raise ValueError("FLUX3 image encoding requires CHW pixels")
    packed = vae.encode(pixels.unsqueeze(0))
    height, width = packed.shape[-2:]
    tokens = packed[0].permute(1, 2, 0).reshape(height * width, packed.shape[1])
    return Flux3ImageLatents(tokens, image_ids(height, width, device=pixels.device), height, width)


def decode_image(vae: Flux2VAE, image: Flux3ImageLatents) -> Tensor:
    """Decode one dense native token grid into CHW pixels with a complete gradient path.

    Require canonical row-major IDs rather than inferring geometry from maxima;
    reject shuffled/duplicate/missing positions before decoding. No implicit ID
    sorting, clipping or resizing occurs. The caller owns VAE placement/mode.
    This boundary is called per image, outside the training microbatch hot path.
    """
    if (
        image.tokens.ndim != 2
        or image.tokens.shape != (image.height * image.width, 4 * vae.config.latent_channels)
        or image.ids.shape != (image.height * image.width, 4)
        or image.ids.dtype != torch.int64
        or image.ids.device != image.tokens.device
        or not torch.equal(image.ids, image_ids(image.height, image.width, device=image.tokens.device))
    ):
        raise ValueError("FLUX3 decoding requires matching tokens and canonical dense row-major IDs")
    packed = image.tokens.transpose(0, 1).reshape(1, -1, image.height, image.width)
    return vae.decode(packed)[0]
