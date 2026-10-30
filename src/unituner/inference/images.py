"""Stage text, native conditioning, image sampling and decoding for shared image families."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from unituner.config.generation import GenerationConfig
from unituner.inference.adapters import image_adapter
from unituner.inference.decoding import decode_on_device
from unituner.inference.publication import publish_images
from unituner.inference.sampling import generate_latents
from unituner.models.catalog import ModelFamily
from unituner.models.components.conditioning import Caption
from unituner.models.components.image_encoders import load_image_text, load_image_vae
from unituner.models.flux.conditioning import FluxTextFeatures
from unituner.models.flux.inference.conditioning import build_flux_condition
from unituner.models.flux.transformer import FluxTransformer
from unituner.models.flux.vae import FluxVAE
from unituner.models.flux2.inference.conditioning import build_flux2_references
from unituner.models.flux2.transformer import Flux2Transformer
from unituner.models.flux2.vae import Flux2VAE
from unituner.models.flux3.inference.conditioning import build_flux3_references
from unituner.models.flux3.transformer import Flux3Transformer
from unituner.models.image import load_image_model
from unituner.models.moss_image.inference.conditioning import encode_references
from unituner.models.moss_image.transformer import MossImageTransformer
from unituner.models.z_image.transformer import ZImageTransformer
from unituner.models.z_image.vae import ZImageVAE

if TYPE_CHECKING:
    from unituner.models.components.image_encoders import PromptTextEncoder
    from unituner.models.flux.text.encoder import FluxTextEncoder


def _encode_prompts(
    config: GenerationConfig, device: torch.device
) -> tuple[Caption | FluxTextFeatures, Caption | None]:
    text = load_image_text(config.model, device=device)
    if config.model.family == ModelFamily.FLUX:
        return cast("FluxTextEncoder", text).encode([config.sampling.prompt]), None
    encoder = cast("PromptTextEncoder", text)
    with torch.autocast(device.type, enabled=False):
        positive = encoder.encode_prompt(config.sampling.prompt)
        needs_negative = (
            config.sampling.negative_prompt is not None
            or config.model.family == ModelFamily.MOSS_IMAGE
            or (config.model.family in (ModelFamily.FLUX2, ModelFamily.FLUX3) and config.model.variant == "base")
        )
        default_negative = " " if config.model.family == ModelFamily.FLUX2 else ""
        negative = (
            encoder.encode_prompt(
                default_negative if config.sampling.negative_prompt is None else config.sampling.negative_prompt
            )
            if needs_negative
            else None
        )
    return positive, negative


def run(config: GenerationConfig, *, device: torch.device) -> int:
    """Release text before denoising and the denoiser before native VAE decoding."""
    if config.model.family not in (
        ModelFamily.FLUX,
        ModelFamily.FLUX2,
        ModelFamily.FLUX3,
        ModelFamily.MOSS_IMAGE,
        ModelFamily.Z_IMAGE,
    ):
        raise ValueError("No shared image generation pipeline for this model family")
    if config.model.vae is None or config.model.text_encoder is None:
        raise ValueError("Generation requires model.vae and model.text_encoder")
    positive, negative = _encode_prompts(config, device)
    vae = load_image_vae(Path(config.model.vae), family=config.model.family, device=torch.device("cpu"))
    references = ()
    if config.model.family == ModelFamily.FLUX2:
        references = build_flux2_references(
            cast(Flux2VAE, vae),
            tuple(Path(path) for path in config.sampling.reference_images),
            original=config.sampling.conditioning_resize == "original",
        )
    elif config.model.family == ModelFamily.FLUX3:
        references = build_flux3_references(
            cast(Flux2VAE, vae), tuple(Path(path) for path in config.sampling.reference_images)
        )
    elif config.model.family == ModelFamily.MOSS_IMAGE:
        references = encode_references(cast(Flux2VAE, vae), config.sampling.reference_images, device=device)
    model = load_image_model(config.model, device=device, dtype=torch.float32).eval()
    condition = None
    if isinstance(model, FluxTransformer):
        condition = build_flux_condition(cast(FluxVAE, vae), config.sampling, variant=model.config.variant)
        channels = model.config.out_channels // 4
    elif isinstance(model, Flux3Transformer):
        channels = model.config.image_channels
    elif isinstance(model, MossImageTransformer):
        channels = model.config.latent_channels
    elif isinstance(model, Flux2Transformer | ZImageTransformer):
        channels = model.config.in_channels
    else:
        raise TypeError("Shared generation requires FLUX, FLUX.2, FLUX3, MOSS or Z-Image")
    vae_channels = vae.config.latent_channels * (4 if isinstance(vae, Flux2VAE) else 1)
    if model.config.variant != config.model.variant or channels != vae_channels:
        raise ValueError("Generation model variant/channels differ from the supplied model/vae")
    factor = 2 ** (len(vae.config.block_channels) - 1) if isinstance(vae, ZImageVAE) else vae.config.spatial_factor
    with torch.autocast(device.type, enabled=False), image_adapter(model, config.adapter):
        latents = generate_latents(
            model,
            positive,
            config.sampling,
            negative=negative,
            references=references,
            condition=condition,
            vae_factor=factor,
        ).cpu()
    del model
    pixels = decode_on_device(vae, latents, device=device)
    return publish_images(
        pixels,
        Path(config.output),
        rounding="truncate" if config.model.family in (ModelFamily.FLUX2, ModelFamily.FLUX3) else "nearest",
    )
