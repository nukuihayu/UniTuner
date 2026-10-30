"""Predict restored latents directly from normalized low-quality image encodings."""

import torch
from torch import Tensor

from unituner.models.components.conditioning import Caption
from unituner.models.z_image.transformer import ZImageTransformer


def restore_latents(model: ZImageTransformer, low_latents: Tensor, caption: Caption, *, sigma: float) -> Tensor:
    """Return low + sigma*prediction with FP32 Euler arithmetic and native time 1-sigma.

    Low latents are nonempty floating BCHW in VAE-normalized model units.
    Sigma is a finite noise fraction strictly inside (0,1). Feed the encoding
    directly, without noise injection or CFG. Preserve gradients to the input,
    encoder adapters and denoiser; no RNG or model mode is changed.
    """
    if low_latents.ndim != 4 or not low_latents.numel() or not low_latents.is_floating_point() or not 0 < sigma < 1:
        raise ValueError("Z-Image restoration requires nonempty floating BCHW latents and 0 < sigma < 1")
    timesteps = torch.full((low_latents.shape[0],), 1 - sigma, device=low_latents.device, dtype=torch.float32)
    prediction = model(low_latents.unsqueeze(2), timesteps, caption.features, caption.mask).squeeze(2)
    if prediction.shape != low_latents.shape:
        raise ValueError("Restoration prediction must match the input latent shape")
    return low_latents.float() + sigma * prediction.float()
