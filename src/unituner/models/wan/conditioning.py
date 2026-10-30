"""Encode native Wan mean latents and first-frame video conditions."""

from dataclasses import dataclass

import torch
from torch import Tensor

from unituner.models.qwen_image.vae import QwenImageVAE


@dataclass(frozen=True, slots=True)
class WanLatentConditions:
    """Own BCTHW mean latents and optional native video-conditioning tensors.

    image_condition contains four first-frame visibility channels followed by
    sixteen encoded image channels. It is a model I2V input, not an RCM loss mask.
    control_latents preserve the control video's own temporal extent.
    """

    latents: Tensor
    image_condition: Tensor | None
    control_latents: Tensor | None


def validate_wan_vae(vae: QwenImageVAE) -> None:
    """Require native shared RGB/16-channel/factor8 Wan causal VAE geometry."""
    config = vae.config
    if (
        config.input_channels != 3
        or config.latent_channels != 16
        or config.channel_multipliers != (1, 2, 4, 4)
        or config.temporal_downsample != (False, True, True)
        or config.attention_scales
    ):
        raise ValueError("Wan conditions require the native RGB/16-channel/factor8 shared vae")


def encode_first_frame_condition(vae: QwenImageVAE, pixels: Tensor) -> Tensor:
    """Encode first-frame plus normalized-zero middle frames as native B20FHW.

    Input is FP32 RGB BCTHW with1+4k frames; only its first frame is used.
    Keep per-row causal histories independent and prepend native four-channel
    visibility. Do not encode a second, unused target video during generation.
    Preserve gradients to the first image and allocate new condition storage.
    """
    validate_wan_vae(vae)
    if pixels.ndim != 5 or min(pixels.shape) < 1 or pixels.shape[1] != 3 or (pixels.shape[2] - 1) % 4:
        raise ValueError("Wan image conditions require RGB BCTHW with1+4k frames")
    images = torch.cat((pixels[:, :, :1], torch.zeros_like(pixels[:, :, 1:])), dim=2)
    latents = torch.cat([vae.encode(row.unsqueeze(0)) for row in images], dim=0)
    batch, _, frames, height, width = latents.shape
    visibility = torch.cat(
        (latents.new_ones(batch, 4, 1, height, width), latents.new_zeros(batch, 4, frames - 1, height, width)), dim=2
    )
    return torch.cat((visibility, latents), dim=1)


def encode_latent_conditions(
    vae: QwenImageVAE,
    pixels: Tensor,
    *,
    image_conditioning: bool = False,
    control_pixels: Tensor | None = None,
) -> WanLatentConditions:
    """Encode FP32 RGB BCTHW pixels in [-1,1] with Wan's mean-only policy.

    The shared VAE owns spatial8/temporal4 causal compression. Each batch row
    is encoded separately, preserving the actual Wan wrapper's convolution
    arithmetic and independent temporal histories. I2V encodes the first frame
    followed by normalized zero pixels, then prepends four visibility channels.
    No stochastic posterior sample or RCM mask enters this operation. Gradients
    remain connected; the caller owns no-grad, frozen weights and residency.
    """
    validate_wan_vae(vae)
    if pixels.ndim != 5 or min(pixels.shape) < 1 or pixels.shape[1] != 3 or (pixels.shape[2] - 1) % 4:
        raise ValueError("Wan video pixels require nonempty RGB BCTHW with1+4k frames")
    if control_pixels is not None and (
        control_pixels.ndim != 5
        or min(control_pixels.shape) < 1
        or control_pixels.shape[:2] != pixels.shape[:2]
        or control_pixels.shape[-2:] != pixels.shape[-2:]
        or (control_pixels.shape[2] - 1) % 4
        or control_pixels.dtype != pixels.dtype
        or control_pixels.device != pixels.device
    ):
        raise ValueError("Wan control video must match RGB batch/spatial/dtype/device and have1+4k frames")
    latents = torch.cat([vae.encode(row.unsqueeze(0)) for row in pixels], dim=0)
    image_condition = encode_first_frame_condition(vae, pixels) if image_conditioning else None
    control_latents = None
    if control_pixels is not None:
        control_latents = torch.cat([vae.encode(row.unsqueeze(0)) for row in control_pixels], dim=0)
    return WanLatentConditions(latents, image_condition, control_latents)
