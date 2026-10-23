"""Restore Krea2 image latents through direct low-quality injection at native model time."""

import torch
from torch import Tensor

from unitunner.models.krea2.conditioning import Krea2TextFeatures, image_positions, pack_latents, unpack_latents
from unitunner.models.krea2.transformer import Krea2Transformer


def restore_latents(
    model: Krea2Transformer, low_latents: Tensor, caption: Krea2TextFeatures, *, sigma: float
) -> Tensor:
    """Return FP32 low-sigma*velocity, preserving encoder/text/model gradients.

    Latents are normalized BC1HW with patch-divisible H/W. Text retains BSLD
    selected layers and host valid-prefix lengths. Feed the low-quality latent
    directly, without extra noising, CFG or RCM mask. Native time follows
    full(sigma*1000)/1000, matching the actual restoration trainer boundary.
    Preserve caller mode, RNG, parameter identity and input storage.
    """
    if (
        low_latents.ndim != 5
        or min(low_latents.shape) < 1
        or low_latents.shape[1:3] != (model.config.in_channels, 1)
        or any(size % model.config.patch_size for size in low_latents.shape[-2:])
        or low_latents.dtype != torch.float32
        or not 0 < sigma < 1
    ):
        raise ValueError("Krea2 restoration requires FP32 BC1HW native channels, patch-divisible H/W and0<sigma<1")
    values = low_latents[:, :, 0]
    timesteps = torch.full((values.shape[0],), sigma * 1000, device=values.device, dtype=torch.float32)
    prediction = model(
        pack_latents(values, patch=model.config.patch_size),
        caption.features,
        timesteps / 1000,
        image_positions(values, patch=model.config.patch_size, text_count=caption.features.shape[1]),
        text_lengths=caption.lengths,
    )
    velocity = unpack_latents(
        prediction, patch=model.config.patch_size, height=values.shape[-2], width=values.shape[-1]
    )[:, :, None]
    return low_latents - sigma * velocity.float()
