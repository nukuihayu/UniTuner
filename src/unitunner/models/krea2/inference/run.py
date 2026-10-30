"""Own the native Krea2 image generation stages and artifact validation."""

from pathlib import Path

import torch

from unitunner.config.generation import GenerationConfig
from unitunner.inference.adapters import image_adapter
from unitunner.inference.decoding import decode_on_device
from unitunner.inference.publication import publish_images
from unitunner.inference.sampling import generate_latents
from unitunner.models.krea2.conditioning import Krea2TextFeatures
from unitunner.models.krea2.inference.conditioning import encode_krea2_prompts
from unitunner.models.krea2.loading import load_krea2
from unitunner.models.krea2.text.encoder import load_krea2_text
from unitunner.models.qwen_image.loading import load_qwen_image_vae


def run(config: GenerationConfig, *, device: torch.device) -> int:
    """Stage native FP32 Krea2 text/denoiser/shared VAE and strictly load published adapters."""
    if config.model.vae is None or config.model.text_encoder is None:
        raise ValueError("Krea2 generation requires vae/text assets")
    text = load_krea2_text(
        Path(config.model.text_encoder),
        device=device,
        dtype=torch.float32,
        config=config.model.text,
        source_format=config.model.text_source_format or "auto",
    )
    positive, negative = encode_krea2_prompts(text, config.sampling, variant=config.model.variant)
    positive = Krea2TextFeatures(positive.features.cpu(), positive.lengths)
    if negative is not None:
        negative = Krea2TextFeatures(negative.features.cpu(), negative.lengths)
    del text
    model = load_krea2(
        Path(config.model.weights), variant=config.model.variant, device=device, dtype=torch.float32
    ).eval()
    if model.config.variant != config.model.variant:
        raise ValueError("Krea2 generation variant differs from artifact")
    vae = load_qwen_image_vae(Path(config.model.vae)).eval().requires_grad_(False)
    if vae.config.input_channels != 3 or vae.config.latent_channels != 16 or vae.config.spatial_factor != 8:
        raise ValueError("Krea2 requires the RGB shared Qwen vae with latent16/factor8")
    with torch.autocast(device.type, enabled=False), image_adapter(model, config.adapter):
        latents = generate_latents(model, positive, config.sampling, negative=negative).cpu()
    del model
    pixels = decode_on_device(vae, latents.unsqueeze(2), device=device)
    return publish_images(pixels, Path(config.output), rounding="truncate")
