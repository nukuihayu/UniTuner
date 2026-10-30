"""Restore FLUX.2 packed VAE latents with native text and embedded guidance."""

import torch
from torch import Tensor

from unituner.models.components.conditioning import Caption
from unituner.models.flux2.conditioning import pack_image, pack_text
from unituner.models.flux2.transformer import Flux2Transformer


def restore_latents(model: Flux2Transformer, low_latents: Tensor, caption: Caption, *, sigma: float) -> Tensor:
    """Return FP32 low-sigma*velocity at full(sigma*1000)/1000, with no extra noise.

    Inputs are BCHW packed-BN VAE latents and native BLD text. References
    are absent; Dev embeds guidance1, Klein has no guidance. Host lengths
    retain ragged text without attending padding. Preserve all gradients/RNG.
    """
    if low_latents.ndim != 4 or low_latents.dtype != torch.float32 or not 0 < sigma < 1:
        raise ValueError("FLUX.2 restoration requires FP32 BCHW and0<sigma<1")
    if caption.mask is not None and caption.lengths is None:
        raise ValueError("FLUX.2 restoration requires host text lengths for ragged captions")
    lengths = caption.lengths or (caption.features.shape[1],) * low_latents.shape[0]
    timesteps = torch.full((low_latents.shape[0],), sigma * 1000, device=low_latents.device, dtype=torch.float32)
    rows = []
    for index, value in enumerate(low_latents):
        tokens, ids = pack_image(value)
        text, text_ids = pack_text(caption.features[index, : lengths[index]])
        guidance = value.new_ones((1,)) if model.config.use_guidance_embed else None
        prediction = model(
            tokens.unsqueeze(0),
            ids.unsqueeze(0),
            timesteps[index : index + 1] / 1000,
            text.unsqueeze(0),
            text_ids.unsqueeze(0),
            guidance,
            target_count=tokens.shape[0],
        )
        rows.append(prediction[0].transpose(0, 1).reshape_as(value))
    return low_latents - sigma * torch.stack(rows).float()
