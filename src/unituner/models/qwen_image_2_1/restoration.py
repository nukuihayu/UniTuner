"""Restore normalized Qwen2.1 image latents by direct LQ velocity injection."""

import torch
from torch import Tensor

from unituner.models.components.conditioning import Caption
from unituner.models.qwen_image_2_1.conditioning import pack_latents, unpack_latents
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer


def restore_latents(model: QwenImage21Transformer, low_latents: Tensor, caption: Caption, *, sigma: float) -> Tensor:
    """Return FP32 low-sigma*velocity with no extra noising, references, CFG or RCM mask.

    Latents use normalized BC1HW, with native patch1 grids. Text uses BLD and
    optional host prefix lengths from batch_captions. Native model time follows
    full(sigma*1000)/1000, matching the actual restoration trainer call_dit
    boundary. Preserve encoder/input/denoiser gradients, modes and caller RNG.
    """
    if (
        low_latents.ndim != 5
        or min(low_latents.shape) < 1
        or low_latents.shape[1:3] != (model.config.in_channels, 1)
        or model.config.out_channels != model.config.in_channels
        or low_latents.dtype != torch.float32
        or not 0 < sigma < 1
        or (caption.mask is not None and caption.lengths is None)
    ):
        raise ValueError("Qwen Image2.1 restoration requires FP32 BC1HW,0<sigma<1 and host caption lengths for padding")
    timesteps = torch.full((low_latents.shape[0],), sigma * 1000, device=low_latents.device, dtype=torch.float32)
    prediction = model(
        pack_latents(low_latents),
        caption.features,
        timesteps / 1000,
        (((1, low_latents.shape[-2], low_latents.shape[-1]),),),
        text_lengths=caption.lengths,
    )
    return (
        low_latents
        - sigma * unpack_latents(prediction, height=low_latents.shape[-2], width=low_latents.shape[-1]).float()
    )
