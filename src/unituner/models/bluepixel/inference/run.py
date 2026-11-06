"""Stage local BluePixel text, DCAE reference encoding, transformer, and decoding."""

from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from unituner.config.generation import GenerationConfig
from unituner.inference.adapters import image_adapter
from unituner.inference.publication import publish_images
from unituner.models.bluepixel.inference.conditioning import read_reference
from unituner.models.bluepixel.inference.sampling import sample_latents
from unituner.models.bluepixel.loading import load_bluepixel
from unituner.models.bluepixel.vae import load_bluepixel_vae
from unituner.models.components.image_encoders import load_image_text
from unituner.models.flux2.text.encoder import Flux2TextEncoder


@torch.no_grad()
def generate_pixels(
    config: GenerationConfig, *, device: torch.device,
) -> Tensor:
    """Generate a BCHW edited image from a local reference."""
    selected = config.model
    if selected.vae is None or selected.text_encoder is None:
        raise ValueError("BluePixel requires model.vae and model.text_encoder")
    text = cast(Flux2TextEncoder, load_image_text(selected, device=device))
    positive = text.encode_prompt(config.sampling.prompt).features[0]
    guidance = 3.0 if config.sampling.guidance is None else config.sampling.guidance
    negative = text.encode_prompt("").features[0] if guidance > 1 else None
    del text

    vae = load_bluepixel_vae(Path(selected.vae))
    reference_pixels = read_reference(
        Path(config.sampling.reference_images[0]), width=config.sampling.width, height=config.sampling.height
    )
    try:
        vae.to(device)
        reference = vae.encode(reference_pixels.to(device)).cpu()
    finally:
        vae.cpu()

    model = load_bluepixel(Path(selected.weights), device=device, dtype=torch.bfloat16).eval()
    if model.config.in_channels != 128:
        raise ValueError("BluePixel transformer and DCAE latent channels differ")
    generator = torch.Generator(device="cpu").manual_seed(config.sampling.seed)
    noise = torch.randn(reference.shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    with torch.autocast(device.type, enabled=False), image_adapter(model, config.adapter):
        latents = sample_latents(
            model, noise.to(device), reference.to(device=device, dtype=torch.bfloat16),
            positive.to(device=device, dtype=torch.bfloat16),
            None if negative is None else negative.to(device=device, dtype=torch.bfloat16),
            steps=config.sampling.steps or 4,
            guidance=guidance,
        ).cpu()
    del model
    try:
        vae.to(device)
        decoded = vae.decode(latents.to(device=device, dtype=torch.float32)).float().cpu()
    finally:
        vae.cpu()
    return decoded


def run(config: GenerationConfig, *, device: torch.device) -> int:
    """Publish an edited image using the standard generation configuration."""
    return publish_images(generate_pixels(config, device=device), Path(config.output), rounding="truncate")
