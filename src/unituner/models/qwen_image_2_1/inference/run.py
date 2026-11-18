"""Own the native Qwen Image 2.1 image generation stages and artifact validation."""

from pathlib import Path

import torch

from unituner.config.generation import GenerationConfig
from unituner.inference.adapters import image_adapter
from unituner.inference.decoding import decode_on_device
from unituner.inference.publication import publish_images
from unituner.inference.sampling import generate_latents
from unituner.models.qwen_image_2_1.inference.conditioning import build_qwen_image_21_conditions
from unituner.models.qwen_image_2_1.loading import load_qwen_image_21, load_qwen_image_21_vae
from unituner.models.qwen_image_2_1.text.encoder import load_qwen_image_21_text


def run(config: GenerationConfig, *, device: torch.device) -> int:
    """Encode conditions on CPU like previews, then stage native device denoising and decoding."""
    if config.model.vae is None or config.model.text_encoder is None:
        raise ValueError("Qwen Image2.1 generation requires vae/text assets")
    vae = load_qwen_image_21_vae(Path(config.model.vae)).eval().requires_grad_(False)
    text = load_qwen_image_21_text(
        Path(config.model.text_encoder),
        device=torch.device("cpu"),
        dtype=torch.float32,
        source_format=config.model.text_source_format or "auto",
        text_config=config.model.text,
    )
    conditions = build_qwen_image_21_conditions(vae, text, config.sampling)
    del text
    model = load_qwen_image_21(Path(config.model.weights), device=device, dtype=torch.float32).eval()
    if (
        model.config.in_channels != vae.config.latent_channels
        or model.config.out_channels != vae.config.latent_channels
    ):
        raise ValueError("Qwen Image2.1 generation model/vae channels differ")
    with torch.autocast(device.type, enabled=False), image_adapter(model, config.adapter):
        latents = generate_latents(
            model,
            conditions.positive,
            config.sampling,
            negative=conditions.negative,
            references=conditions.references,
            inpainting_mask=conditions.mask,
            vae_factor=vae.config.spatial_factor,
        ).cpu()
    del model
    pixels = decode_on_device(vae, latents, device=device)
    return publish_images(pixels, Path(config.output), rounding="truncate")
