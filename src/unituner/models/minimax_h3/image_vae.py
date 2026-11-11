"""Native single-frame H3-Image encoder and 2D ViT decoder."""

from pathlib import Path

import torch
from torch import Tensor, nn

from unituner.models.components.vae_loading import load_vae_state, vae_tensors
from unituner.models.minimax_h3.config.video_vae import MiniMaxH3VideoDecoderConfig, MiniMaxH3VideoVAEConfig
from unituner.models.minimax_h3.vae import LATENTS_MEAN, LATENTS_STD, MiniMaxH3VideoVAE, VideoDecoder

_CPU = torch.device("cpu")


def image_vae_config() -> MiniMaxH3VideoVAEConfig:
    """Return the released image VAE topology (shared H3 encoder, 2D decoder)."""
    return MiniMaxH3VideoVAEConfig(
        token_drop=0,
        decoder=MiniMaxH3VideoDecoderConfig(num_layers=24, num_heads=16, rope_dim_ratio=0.5, rope_dims=2),
    )


class MiniMaxH3ImageVAE(MiniMaxH3VideoVAE):
    """Encode and decode one frame; decoding never repeats temporal tokens."""

    def __init__(self, config: MiniMaxH3VideoVAEConfig | None = None) -> None:
        config = image_vae_config() if config is None else config
        super().__init__(config, include_decoder=False)
        self.post_quant_conv = nn.Conv2d(24, 24, kernel_size=1)
        decoder = config.decoder
        self.decoder = VideoDecoder(
            patch_size=self.vae_ratio,
            patch_size_t=1,
            num_layers=decoder.num_layers,
            heads=decoder.num_heads,
            dim_head=decoder.head_dim,
            rope_theta=decoder.rope_theta,
            rope_dim_ratio=decoder.rope_dim_ratio,
            rope_dims=2,
            bias=decoder.bias,
            eps=decoder.eps,
            num_register_tokens=decoder.num_register_tokens,
        )

    def encode(self, pixels: Tensor, *, posterior_noise: Tensor | None = None) -> Tensor:
        """Use the released fixed posterior seed for reference images by default."""
        if posterior_noise is None:
            posterior = self.posterior(pixels)
            generator = torch.Generator(device="cpu").manual_seed(42)
            posterior_noise = torch.randn(posterior.mean.shape, generator=generator, dtype=torch.float32).to(
                posterior.mean.device
            )
            return self.normalize(posterior.sample(posterior_noise))
        return super().encode(pixels, posterior_noise=posterior_noise)

    def _decode_clip(self, latents: Tensor) -> Tensor:
        """Apply the image checkpoint's 2D post-quant projection to each spatial tile."""
        if self.decoder is None or self.post_quant_conv is None:
            raise RuntimeError("H3-Image VAE decoder is unavailable")

        def decode_tile(tile: Tensor) -> Tensor:
            return self.decoder(self.post_quant_conv(tile.squeeze(2)).unsqueeze(2))

        if not self.use_tiling:
            return decode_tile(latents)
        height_starts, height_lengths, height_overlaps = self._split_tiles(latents.shape[-2] * self.vae_ratio)
        width_starts, width_lengths, width_overlaps = self._split_tiles(latents.shape[-1] * self.vae_ratio)
        tiles = [
            [
                decode_tile(
                    latents[
                        ...,
                        top // self.vae_ratio : (top + height) // self.vae_ratio,
                        left // self.vae_ratio : (left + width) // self.vae_ratio,
                    ]
                )
                for left, width in zip(width_starts, width_lengths, strict=True)
            ]
            for top, height in zip(height_starts, height_lengths, strict=True)
        ]
        return self._stitch_tiles(tiles, height_overlaps, width_overlaps)

    def decode(self, latents: Tensor) -> Tensor:
        """Decode a normalized B24x1xHxW latent directly to one pixel frame."""
        self._validate_values(latents, pixels=False)
        if latents.shape[2] != 1:
            raise ValueError("H3-Image VAE decodes one latent frame")
        values = latents * self.latents_std.view(1, -1, 1, 1, 1).to(latents)
        values = values + self.latents_mean.view(1, -1, 1, 1, 1).to(latents)
        pixels = self._decode_clip(values)
        pixels = pixels.float() * self.pixel_std.to(pixels) + self.pixel_mean.to(pixels)
        return pixels.clamp(0, 1) * 2 - 1


def load_image_vae(
    path: Path,
    *,
    config: MiniMaxH3VideoVAEConfig | None = None,
    device: torch.device = _CPU,
) -> MiniMaxH3ImageVAE:
    """Load the released native VAE checkpoint with strict tensor inventory."""
    with torch.device("meta"):
        model = MiniMaxH3ImageVAE(config)
    tensors = dict(vae_tensors(path, prefixes=("first_stage_model.", "vae.")))
    tensors.setdefault("latents_mean", torch.tensor(LATENTS_MEAN))
    tensors.setdefault("latents_std", torch.tensor(LATENTS_STD))
    load_vae_state(model, tensors, device=device, label="MiniMax-H3-Image VAE", nonnegative=("latents_std",))
    model.initialize_nonpersistent_buffers(device)
    return model
