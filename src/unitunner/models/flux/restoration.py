"""Restore FLUX image latents with direct LQ injection and native pooled text."""

import torch
from torch import Tensor

from unitunner.models.flux.conditioning import FluxTextFeatures, image_ids, pack_latents, unpack_latents
from unitunner.models.flux.transformer import FluxTransformer


def restore_latents(model: FluxTransformer, low_latents: Tensor, caption: FluxTextFeatures, *, sigma: float) -> Tensor:
    """Return FP32 low-sigma*velocity at full(sigma*1000)/1000, without noising or CFG.

    Latents are normalized BCHW with even H/W; T5 BLD and pooled CLIP BV
    remain active. Only text-to-image variants are accepted. Preserve all
    encoder/denoiser/text gradients, modes, storage and caller RNG. The learned
    guidance embedding is fixed at 1.0, matching Trainer's OMGSR-F path; this
    is distinct from the 3.5 flow-matching/sampling default and from CFG.
    """
    if low_latents.ndim != 4 or low_latents.dtype != torch.float32 or not 0 < sigma < 1:
        raise ValueError("FLUX restoration requires FP32 BCHW and0<sigma<1")
    if model.config.variant != "dev":
        raise ValueError("FLUX restoration requires the text-to-image dev variant")
    batch, _, height, width = low_latents.shape
    timesteps = torch.full((batch,), sigma * 1000, device=low_latents.device, dtype=torch.float32)
    guidance = low_latents.new_full((batch,), 1.0) if model.config.guidance_embed else None
    prediction = model(
        pack_latents(low_latents),
        image_ids(batch, height // 2, width // 2, device=low_latents.device),
        caption.captions,
        low_latents.new_zeros((*caption.captions.shape[:2], 3)),
        timesteps / 1000,
        caption.pooled,
        guidance,
    )
    return low_latents - sigma * unpack_latents(prediction, height, width).float()
