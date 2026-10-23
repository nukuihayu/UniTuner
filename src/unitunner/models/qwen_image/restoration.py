"""Predict one restored RGB latent using Qwen Image's native velocity convention."""

import torch
from torch import Tensor

from unitunner.models.components.conditioning import Caption
from unitunner.models.qwen_image.conditioning import pack_latents, unpack_latents
from unitunner.models.qwen_image.transformer import QwenImageTransformer


def restore_latents(model: QwenImageTransformer, low_latents: Tensor, caption: Caption, *, sigma: float) -> Tensor:
    """Return low - sigma*velocity in FP32 with direct normalized-time LQ injection.

    Require original Qwen Image and normalized BC1HW image latents with even
    spatial extents. Feed LQ directly without noising, references or CFG. Native
    prediction is noise-minus-data. Preserve input/encoder/denoiser gradients,
    caller RNG and model modes. Caption validity uses the native boolean BL mask.
    """
    if (
        model.config.variant != "original"
        or low_latents.ndim != 5
        or min(low_latents.shape) < 1
        or low_latents.shape[1:3] != (model.config.out_channels, 1)
        or any(size % 2 for size in low_latents.shape[-2:])
        or not low_latents.is_floating_point()
        or not 0 < sigma < 1
    ):
        raise ValueError("Qwen restoration requires original, floating BC1HW native latents with even H/W and0<sigma<1")
    timesteps = torch.full((low_latents.shape[0],), sigma, device=low_latents.device, dtype=torch.float32)
    prediction = model(
        pack_latents(low_latents),
        caption.features,
        timesteps,
        ((1, low_latents.shape[-2] // 2, low_latents.shape[-1] // 2),),
        text_mask=caption.mask,
    )
    prediction = unpack_latents(prediction, layers=1, height=low_latents.shape[-2], width=low_latents.shape[-1])
    return low_latents.float() - sigma * prediction.float()
