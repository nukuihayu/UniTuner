"""Describe transformer-visible VAE geometry at the shared image boundary."""

from dataclasses import dataclass

from unituner.models.flux.vae import FluxVAE
from unituner.models.flux2.vae import Flux2VAE
from unituner.models.qwen_image.vae import QwenImageVAE
from unituner.models.qwen_image_2_1.vae import QwenImage21VAE
from unituner.models.z_image.vae import ZImageVAE


@dataclass(frozen=True, slots=True)
class ImageLatentGeometry:
    """Pin effective pixel stride, model channels and restoration padding independently."""

    spatial_factor: int
    channels: int
    restoration_alignment: int


def image_latent_geometry(
    vae: FluxVAE | Flux2VAE | QwenImageVAE | QwenImage21VAE | ZImageVAE,
) -> ImageLatentGeometry:
    """Keep normalization/packing inside the VAE and expose only effective geometry.

    FLUX.2 includes its 2x2 channel packing in the effective stride/channels.
    Preserve the existing restoration padding for native VAEs.
    """
    factor = 2 ** (len(vae.config.block_channels) - 1) if isinstance(vae, ZImageVAE) else vae.config.spatial_factor
    channels = vae.config.latent_channels * (4 if isinstance(vae, Flux2VAE) else 1)
    return ImageLatentGeometry(factor, channels, 2 * factor)
