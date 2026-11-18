"""Restore FLUX3 packed image latents using the native three-mode transformer."""

import torch
from torch import Tensor

from unituner.models.components.conditioning import Caption
from unituner.models.flux3.conditioning import Flux3Sample, image_tokens, pack_samples, text_tokens
from unituner.models.flux3.transformer import Flux3Transformer


def restore_latents(model: Flux3Transformer, low_latents: Tensor, caption: Caption, *, sigma: float) -> Tensor:
    """Return FP32 low-sigma*velocity with native empty-reference handling.

    Use full(sigma*1000)/1000 for source time and the same destination for
    distilled finetuning, matching the original training boundary. BCHW packed
    VAE latents enter directly; LD captions exclude padded rows via host
    lengths. No random noise, CFG, RCM masks or caller mutation occurs.
    """
    if low_latents.ndim != 4 or low_latents.dtype != torch.float32 or not 0 < sigma < 1:
        raise ValueError("FLUX3 restoration requires FP32 BCHW and0<sigma<1")
    if caption.mask is not None and caption.lengths is None:
        raise ValueError("FLUX3 restoration requires host text lengths for ragged captions")
    lengths = caption.lengths or (caption.features.shape[1],) * low_latents.shape[0]
    timesteps = torch.full((low_latents.shape[0],), sigma * 1000, device=low_latents.device, dtype=torch.float32) / 1000
    samples = [
        Flux3Sample(image_tokens(value), text_tokens(caption.features[index, : lengths[index]]), None)
        for index, value in enumerate(low_latents)
    ]
    prediction = model(pack_samples(samples, timesteps), timesteps if model.config.variant == "distilled" else None)
    velocity = prediction.reshape(low_latents.shape[0], *low_latents.shape[-2:], low_latents.shape[1]).permute(
        0, 3, 1, 2
    )
    return low_latents - sigma * velocity.float()
