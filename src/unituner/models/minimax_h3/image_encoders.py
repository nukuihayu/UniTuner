"""Load the H3-Image VAE and text encoder for cache, previews and inference."""

from pathlib import Path

import torch

from unituner.config.model import EncoderModelConfig
from unituner.models.minimax_h3.image_vae import MiniMaxH3ImageVAE, load_image_vae
from unituner.models.minimax_h3.text.encoder import MiniMaxH3TextEncoder
from unituner.models.minimax_h3.text.loading import load_text_encoder


def load_image_encoders(
    model: EncoderModelConfig, *, device: torch.device, text_device: torch.device | None = None
) -> tuple[MiniMaxH3ImageVAE, MiniMaxH3TextEncoder]:
    """Honor the configured text device while keeping the image VAE independently placed."""
    settings = model.minimax_h3
    if settings is None or model.vae is None or model.text_encoder is None:
        raise ValueError("MiniMax-H3-Image requires VAE, text encoder and model.minimax_h3.text_config")
    vae = load_image_vae(Path(model.vae), config=settings.video_topology, device=device).eval().requires_grad_(False)
    text = load_text_encoder(
        Path(model.text_encoder),
        config_directory=Path(settings.text_config),
        device=torch.device(settings.text_device) if settings.text_device else text_device or device,
        source_format=model.text_source_format or "auto",
        text_config=model.text,
        processor_family="minimax-h3-image",
    )
    return vae, text
