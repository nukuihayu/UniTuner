"""Decode native image layouts and scope temporary VAE device residency."""

import torch
from torch import Tensor

from unituner.models.flux.vae import FluxVAE
from unituner.models.flux2.vae import Flux2VAE
from unituner.models.qwen_image.vae import QwenImageVAE
from unituner.models.qwen_image_2_1.vae import QwenImage21VAE
from unituner.models.z_image.vae import ZImageVAE


@torch.no_grad()
def decode_latents(vae: ZImageVAE | Flux2VAE | FluxVAE | QwenImageVAE | QwenImage21VAE, latents: Tensor) -> Tensor:
    """Decode native model latents without changing VAE residency or mode.

    Args:
        VAE: Caller-owned image VAE on its current device and dtype.
        latents: Floating BCHW packed/model-normalized latents for Z-Image/FLUX,
            or VAE-native BC1HW image layers for Qwen. Input values are copied
            to the VAE device/dtype without in-place mutation.

    Returns:
        Owned CPU float32 BCHW RGB/RGBA pixels in native units, without clipping
        or gradients. Qwen layers are flattened into the batch axis; Qwen2.1's
        singleton temporal axis is removed. Z-Image normalization is inverted;
        FLUX and Qwen VAEs own their native latent conversion.
    """
    parameter = next(vae.parameters())
    values = latents.to(device=parameter.device, dtype=parameter.dtype)
    if isinstance(vae, QwenImage21VAE):
        return vae.decode(values)[:, :, 0].float().cpu()
    if isinstance(vae, QwenImageVAE):
        pixels = vae.decode_layers(values)
        return pixels.transpose(1, 2).reshape(-1, vae.config.input_channels, *pixels.shape[-2:]).float().cpu()
    return vae.decode(values if isinstance(vae, Flux2VAE | FluxVAE) else vae.denormalize(values)).float().cpu()


@torch.no_grad()
def decode_on_device(
    vae: ZImageVAE | Flux2VAE | FluxVAE | QwenImageVAE | QwenImage21VAE,
    latents: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    """Decode model-native latents and park the borrowed VAE on CPU, including on failure.

    Args:
        VAE: Frozen evaluation VAE initially on CPU. Its parameter identity,
            dtype and training flags remain unchanged; the caller owns its lifetime.
        latents: Floating native BCHW or BC1HW model latents as required by VAE.
            Values are copied to the VAE device/dtype without changing inputs.
        device: Device used only for this decode. Placement failure also triggers
            cleanup, including when a module transfer completed only partially.

    Returns:
        Owned CPU float32 BCHW pixels in native units, with no gradient graph.
        RGBA and multi-layer outputs follow decode_latents; no clamp or rounding.
    """
    try:
        vae.to(device)
        with torch.autocast(device.type, enabled=False):
            return decode_latents(vae, latents)
    finally:
        vae.cpu()
