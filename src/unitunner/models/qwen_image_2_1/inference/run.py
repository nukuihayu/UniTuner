"""Own the native Qwen Image 2.1 image generation stages and artifact validation."""

from contextlib import ExitStack
from pathlib import Path

import torch

from unitunner.adapters.loading import LoadedAdapter
from unitunner.artifacts.formats.adapters import adapter_prefixes
from unitunner.config.generation import GenerationConfig
from unitunner.inference.decoding import decode_on_device
from unitunner.inference.publication import publish_images
from unitunner.inference.sampling import generate_latents
from unitunner.models.qwen_image_2_1.inference.conditioning import build_qwen_image_21_conditions
from unitunner.models.qwen_image_2_1.loading import load_qwen_image_21, load_qwen_image_21_vae
from unitunner.models.qwen_image_2_1.projections import adapter_projections as qwen21_adapter_projections
from unitunner.models.qwen_image_2_1.text.encoder import load_qwen_image_21_text


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
    adapter = None
    targets = None
    with ExitStack() as owners:
        owners.enter_context(torch.autocast(device.type, enabled=False))
        if config.adapter is not None:
            targets = qwen21_adapter_projections(model)
            adapter = LoadedAdapter(
                Path(config.adapter.weights),
                config.adapter.config,
                targets,
                external_prefixes=adapter_prefixes(targets, algorithm=config.adapter.config.algorithm),
            )
            owners.callback(adapter.close)
        latents = generate_latents(
            model,
            conditions.positive,
            config.sampling,
            negative=conditions.negative,
            references=conditions.references,
            inpainting_mask=conditions.mask,
            vae_factor=vae.config.spatial_factor,
        ).cpu()
    if config.adapter is not None:
        del adapter, targets
    del model
    pixels = decode_on_device(vae, latents, device=device)
    return publish_images(pixels, Path(config.output), rounding="truncate")
