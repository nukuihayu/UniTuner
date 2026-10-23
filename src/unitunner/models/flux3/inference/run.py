"""Own the native FLUX3 image generation stages and artifact validation."""

from contextlib import ExitStack
from pathlib import Path

import torch

from unitunner.adapters.loading import LoadedAdapter
from unitunner.artifacts.formats.adapters import adapter_prefixes
from unitunner.config.generation import GenerationConfig
from unitunner.inference.decoding import decode_on_device
from unitunner.inference.publication import publish_images
from unitunner.inference.sampling import generate_latents
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux2.loading import load_flux2_vae
from unitunner.models.flux3.inference.conditioning import build_flux3_references
from unitunner.models.flux3.loading import load_flux3
from unitunner.models.flux3.projections import adapter_projections as flux3_adapter_projections
from unitunner.models.flux3.text.encoder import load_flux3_text


def run(config: GenerationConfig, *, device: torch.device) -> int:
    """Generate a PNG using local artifacts, releasing each GPU stage before the next.

    Use FP32 computation and family-native seed-local noise. Text features move
    to CPU before the text model is released; denoiser latents likewise move to
    CPU before VAE placement. No unrelated GPU allocations or global RNG are
    reset. Optional published-key adapters use strict explicit rank/algorithm.
    """
    vae_path, text_path = (config.model.vae, config.model.text_encoder)
    if vae_path is None or text_path is None:
        raise ValueError("Generation requires model.vae and model.text_encoder")
    negative: Caption | None = None
    text = load_flux3_text(Path(text_path), device=device, dtype=torch.float32, text_config=config.model.text)
    with torch.autocast(device.type, enabled=False):
        positive = text.encode_prompt(config.sampling.prompt)
        if config.model.variant == "base" or config.sampling.negative_prompt is not None:
            negative = text.encode_prompt(
                "" if config.sampling.negative_prompt is None else config.sampling.negative_prompt
            )
    del text
    vae = load_flux2_vae(Path(vae_path)).eval().requires_grad_(False)
    references = build_flux3_references(vae, tuple(Path(path) for path in config.sampling.reference_images))
    model = load_flux3(
        Path(config.model.weights), variant=config.model.variant, device=device, dtype=torch.float32
    ).eval()
    channels = model.config.image_channels
    vae_channels = 4 * vae.config.latent_channels
    if model.config.variant != config.model.variant or channels != vae_channels:
        raise ValueError("Generation model variant/channels differ from the supplied model/vae")
    adapter = None
    targets = None
    with ExitStack() as stack:
        stack.enter_context(torch.autocast(device.type, enabled=False))
        if config.adapter is not None:
            targets = flux3_adapter_projections(model)
            adapter = LoadedAdapter(
                Path(config.adapter.weights),
                config.adapter.config,
                targets,
                external_prefixes=adapter_prefixes(
                    targets, algorithm=config.adapter.config.algorithm, namespace="diffusion_model"
                ),
            )
            stack.callback(adapter.close)
        latents = generate_latents(
            model,
            positive,
            config.sampling,
            negative=negative,
            references=references,
            vae_factor=vae.config.spatial_factor,
        ).cpu()
    if config.adapter is not None:
        del adapter, targets
    del model
    pixels = decode_on_device(vae, latents, device=device)
    return publish_images(pixels, Path(config.output), rounding="truncate")
