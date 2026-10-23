"""Own native H3 video cache posterior seeds and condition sample rounding."""

import hashlib

import torch
from torch import Tensor

from unitunner.models.minimax_h3.vae import MiniMaxH3VideoVAE


def target_video_seed(cache_seed: int, canonical_item_key: str) -> int:
    """Hash the native UTF8 seed/NUL/item identity into a 63-bit CPU RNG seed."""
    if type(cache_seed) is not int or not isinstance(canonical_item_key, str) or not canonical_item_key:
        raise ValueError("H3 target posterior requires an integer cache seed and nonempty canonical item key")
    digest = hashlib.sha256(f"{cache_seed}\0{canonical_item_key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


@torch.no_grad()
def encode_target_video(vae: MiniMaxH3VideoVAE, pixels: Tensor, *, cache_seed: int, canonical_item_key: str) -> Tensor:
    """Encode native target samples with a local CPU RNG independent of global state.

    Noise is drawn in FP32 on CPU and copied to raw moments before normalization.
    Target samples retain FP32; callers own pixel preprocessing and residency.
    """
    generator = torch.Generator(device="cpu").manual_seed(target_video_seed(cache_seed, canonical_item_key))
    posterior = vae.posterior(pixels)
    noise = torch.randn(posterior.mean.shape, generator=generator, dtype=torch.float32, device="cpu").to(
        posterior.mean.device
    )
    return vae.normalize(posterior.sample(noise))
