"""Bind differentiable RGB restoration images to native normalized VAE layouts."""

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from unitunner.models.flux.loading import load_flux_vae
from unitunner.models.flux.vae import FluxVAE
from unitunner.models.flux2.loading import load_flux2_vae
from unitunner.models.flux2.vae import Flux2VAE
from unitunner.models.qwen_image.loading import load_qwen_image_vae
from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.models.qwen_image_2_1.loading import load_qwen_image_21_vae
from unitunner.models.qwen_image_2_1.vae import QwenImage21VAE
from unitunner.models.z_image.loading import load_z_image_vae
from unitunner.models.z_image.vae import ZImageVAE

type RestorationVAE = ZImageVAE | QwenImageVAE | QwenImage21VAE | FluxVAE | Flux2VAE


def restoration_posterior(family: str, posterior: Literal["mean", "sample"] | None) -> Literal["mean", "sample"]:
    """Resolve the shared training/calibration posterior without drawing noise.

    FLUX.1 and Z-Image follow Trainer's sampled posterior. Qwen/Krea2 retain
    sampling. Packed FLUX.2/FLUX3/MOSS VAEs expose means.
    Previews and restoration inference choose means independently.
    """
    selected = posterior or (
        "sample" if family in ("z-image", "flux", "qwen-image", "qwen-image-2-1", "krea2") else "mean"
    )
    if family in ("flux2", "flux3", "moss-image") and selected != "mean":
        raise ValueError("FLUX.2/FLUX3/MOSS restoration uses posterior mean")
    return selected


def load_restoration_vae(path: Path, *, family: str, device: torch.device) -> RestorationVAE:
    """Load an FP32 frozen native image VAE without text or denoiser weights.

    Family selects exact VAE geometry/normalization. The caller owns the
    returned eval module and its device storage; no adapters are installed.
    Unsupported families fail before artifact IO.
    """
    match family:
        case "z-image":
            vae = load_z_image_vae(path, device=device)
        case "flux":
            vae = load_flux_vae(path, device=device)
        case "flux2" | "flux3" | "moss-image":
            vae = load_flux2_vae(path, device=device)
        case "qwen-image" | "krea2":
            vae = load_qwen_image_vae(path, device=device)
        case "qwen-image-2-1":
            vae = load_qwen_image_21_vae(path, device=device)
        case _:
            raise ValueError("Mid-timestep estimation requires a supported image restoration family")
    return vae.eval().requires_grad_(False)


def encode_restoration_image(
    vae: ZImageVAE | QwenImageVAE | QwenImage21VAE | FluxVAE | Flux2VAE,
    pixels: Tensor,
    *,
    posterior: Literal["mean", "sample"] = "mean",
) -> Tensor:
    """Encode RGB BCHW [-1,1] into normalized BCHW or Qwen BC1HW latents.

    Qwen2.1 RGBA VAEs receive opaque alpha; this is not a restoration mask.
    Sampling draws one matching raw posterior noise tensor from the caller's
    device RNG in logical shape order. FLUX restoration follows Trainer's
    Diffusers log-variance clamp [-30,20], independently of its native cache VAE.
    Mean encoding consumes no RNG. Preserve pixel/encoder graphs;
    Qwen restoration divides by std directly, independently of reciprocal
    scaling in image caches. Caller owns autocast, precision and residency.
    """
    if pixels.ndim != 4 or pixels.shape[1] != 3 or posterior not in ("mean", "sample"):
        raise ValueError("Restoration encoding requires RGB BCHW and mean/sample posterior")
    if isinstance(vae, Flux2VAE):
        if posterior != "mean":
            raise ValueError("Packed FLUX.2/FLUX3/MOSS vaes use posterior mean")
        return vae.encode(pixels)
    if isinstance(vae, QwenImage21VAE) and vae.config.input_channels == 4:
        pixels = torch.cat((pixels, torch.ones_like(pixels[:, :1])), dim=1)
    moments = (
        vae.posterior(pixels)
        if isinstance(vae, FluxVAE)
        else vae.posterior(pixels.unsqueeze(2))
        if isinstance(vae, QwenImageVAE | QwenImage21VAE)
        else vae.encode(pixels)
    )
    if posterior == "sample":
        # Trainer/Diffusers sample logical shape order. Matching strides with
        # randn_like assigns different draws to channels-last posterior elements.
        noise = torch.randn(moments.mean.shape, device=moments.mean.device, dtype=moments.mean.dtype)
        raw = (
            moments.mean + torch.exp(0.5 * moments.log_variance.clamp(-30, 20)) * noise
            if isinstance(vae, FluxVAE)
            else moments.sample(noise)
        )
    else:
        raw = moments.mean
    if isinstance(vae, QwenImageVAE):
        # Restoration uses direct std division; cache/sampling use reciprocal
        # scaling and must keep their independently validated arithmetic order.
        mean = raw.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = raw.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        return (raw - mean) / std
    return vae.normalize(raw)


def decode_restoration_latents(
    vae: ZImageVAE | QwenImageVAE | QwenImage21VAE | FluxVAE | Flux2VAE, latents: Tensor
) -> Tensor:
    """Decode normalized native image latents to RGB BCHW without detaching gradients."""
    if isinstance(vae, FluxVAE | Flux2VAE):
        return vae.decode(latents)
    if isinstance(vae, QwenImage21VAE):
        return vae.decode(latents)[:, :3, 0]
    if isinstance(vae, QwenImageVAE):
        if latents.ndim != 5 or latents.shape[2] != 1 or vae.config.input_channels != 3:
            raise ValueError("Qwen restoration decoding requires RGB vae and BC1HW latents")
        mean = latents.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latents.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        return vae.decode_raw(latents * std + mean)[:, :, 0]
    return vae.decode(vae.denormalize(latents))
