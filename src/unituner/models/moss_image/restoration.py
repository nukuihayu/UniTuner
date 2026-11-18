"""Restore MOSS Image Base/SFT packed VAE latents at native model time."""

import torch
from torch import Tensor

from unituner.models.components.conditioning import Caption
from unituner.models.moss_image.transformer import MossImageTransformer


def restore_latents(model: MossImageTransformer, low_latents: Tensor, caption: Caption, *, sigma: float) -> Tensor:
    """Return FP32 low-sigma*velocity, preserving encoder and transformer gradients.

    Feed normalized packed-BN BCHW latents directly at native full(sigma*1000)
    time. Keep native text validity; do not inject noise or apply CFG. MOSS
    predicts noise-minus-clean velocity. Preserve caller modes/storage/RNG.
    """
    if low_latents.ndim != 4 or low_latents.dtype != torch.float32 or not 0 < sigma < 1:
        raise ValueError("MOSS restoration requires FP32 BCHW and0<sigma<1")
    timesteps = torch.full((low_latents.shape[0],), sigma * 1000, device=low_latents.device, dtype=torch.float32)
    return low_latents - sigma * model(low_latents, timesteps, caption.features, caption.mask).float()
