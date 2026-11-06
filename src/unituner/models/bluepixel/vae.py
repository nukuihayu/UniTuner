"""Adapt the released DCAE-F64 to UniTuner's normalized latent contract."""

from pathlib import Path

import torch
from torch import Tensor, nn

_CPU = torch.device("cpu")


class BluePixelVAE(nn.Module):
    """Own a frozen Diffusers DCAE and its published latent scaling factor."""

    spatial_factor = 64

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        config = vae.config
        if (
            config.latent_channels != 128
            or len(config.encoder_block_out_channels) != 7
            or abs(config.scaling_factor - 0.4538) > 1e-6
        ):
            raise ValueError("BluePixel requires a DCAE-F64 with 128 latent channels")
        self.vae = vae
        self.scaling_factor: float = float(config.scaling_factor)

    @property
    def config(self):
        """Expose the DCAE geometry to shared restoration and cache consumers."""
        return self.vae.config

    @property
    def encoder(self) -> nn.Module:
        """Borrow DCAE's encoder for the existing reversible adapter owner."""
        return self.vae.encoder

    @property
    def decoder(self) -> nn.Module:
        """Expose the same decoder boundary used by the shared VAE execution owner."""
        return self.vae.decoder

    def encode(self, pixels: Tensor) -> Tensor:
        """Return scaled BCHW latents for RGB pixels in [-1,1]."""
        if pixels.ndim != 4 or pixels.shape[1] != 3 or any(size % 64 for size in pixels.shape[-2:]):
            raise ValueError("BluePixel VAE requires BCHW RGB pixels with 64-aligned dimensions")
        return self.vae.encode(pixels, return_dict=False)[0] * self.scaling_factor

    def decode(self, latents: Tensor) -> Tensor:
        """Decode the same scaled latent layout without modifying the caller's tensor."""
        if latents.ndim != 4 or latents.shape[1] != 128:
            raise ValueError("BluePixel VAE requires BCHW latents with 128 channels")
        return self.vae.decode(latents / self.scaling_factor, return_dict=False)[0]

    def enable_tiling(self) -> None:
        """Limit DCAE activation memory at high resolutions."""
        self.vae.enable_tiling()


def load_bluepixel_vae(path: Path, *, device: torch.device = _CPU) -> BluePixelVAE:
    """Load only a local Diffusers DCAE component; never fetch remote code or weights."""
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("BluePixel model.vae must be a local DCAE component directory")
    from diffusers import AutoencoderDC

    vae = AutoencoderDC.from_pretrained(str(path), local_files_only=True, use_safetensors=True)
    return BluePixelVAE(vae).to(device=device, dtype=torch.float32).eval().requires_grad_(False)
