"""Own the native FLUX image generation stages and artifact validation."""

from contextlib import ExitStack
from pathlib import Path

import torch

from unitunner.adapters.loading import LoadedAdapter
from unitunner.artifacts.formats.flux_adapters import flux_adapter_layout
from unitunner.config.generation import GenerationConfig
from unitunner.inference.decoding import decode_on_device
from unitunner.inference.publication import publish_images
from unitunner.inference.sampling import generate_latents
from unitunner.models.flux.inference.conditioning import build_flux_condition
from unitunner.models.flux.loading import load_flux, load_flux_vae
from unitunner.models.flux.projections import adapter_projections as flux_adapter_projections
from unitunner.models.flux.text.encoder import load_flux_text


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

    flux_text = load_flux_text(
        Path(text_path),
        config.model.asset_path("clip_text"),
        device=device,
        max_length=config.model.text.resolved_max_length("flux"),
        text_config=config.model.text,
    )
    positive = flux_text.encode([config.sampling.prompt])
    del flux_text
    condition = None
    vae = load_flux_vae(Path(vae_path)).eval().requires_grad_(False)
    model = load_flux(Path(config.model.weights), variant=config.model.variant, device=device).eval()
    condition = build_flux_condition(vae, config.sampling, variant=model.config.variant)
    channels = model.config.out_channels // 4
    vae_channels = vae.config.latent_channels
    if model.config.variant != config.model.variant or channels != vae_channels:
        raise ValueError("Generation model variant/channels differ from the supplied model/vae")
    adapter = None
    targets = None
    with ExitStack() as stack:
        stack.enter_context(torch.autocast(device.type, enabled=False))
        if config.adapter is not None:
            targets = flux_adapter_projections(model)
            adapter = LoadedAdapter(
                Path(config.adapter.weights),
                config.adapter.config,
                targets,
                projection_layout=flux_adapter_layout(
                    targets, hidden_size=model.config.hidden_size, algorithm=config.adapter.config.algorithm
                ),
            )
            stack.callback(adapter.close)
        latents = generate_latents(
            model, positive, config.sampling, condition=condition, vae_factor=vae.config.spatial_factor
        ).cpu()
    if config.adapter is not None:
        del adapter, targets
    del model
    pixels = decode_on_device(vae, latents, device=device)
    return publish_images(pixels, Path(config.output), rounding="nearest")
