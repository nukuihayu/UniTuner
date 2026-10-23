"""Publish native FLUX previews without owning training state."""

from pathlib import Path

import torch

from unitunner.config.sampling import SamplingConfig
from unitunner.config.training import TrainingConfig
from unitunner.data.media.image import save_image
from unitunner.inference.decoding import decode_on_device
from unitunner.inference.sampling import generate_latents
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux.conditioning import FluxTextFeatures
from unitunner.models.flux.inference.conditioning import build_flux_condition
from unitunner.models.flux.loading import load_flux_vae
from unitunner.models.flux.text.encoder import load_flux_text
from unitunner.models.flux.transformer import FluxTransformer
from unitunner.training.previews.publication import CaptionCache, publish_images


class ImagePreviews(CaptionCache[tuple[tuple[FluxTextFeatures, Caption | None], ...]]):
    """Borrow the live FLUX model and cache CPU text features between previews.

    Construction performs no IO. The caller owns eval/no-grad/RNG restoration;
    each call parks its VAE on CPU even when generation or publication fails.
    """

    def __init__(self, config: TrainingConfig, model: FluxTransformer) -> None:
        if config.model.family != "flux":
            raise ValueError("Preview configuration and native model family differ")
        if config.model.variant != model.config.variant:
            raise ValueError("Preview configuration and artifact variant differ")
        self.config = config
        self.model = model
        self.captions: tuple[tuple[FluxTextFeatures, Caption | None], ...] | None = None

    def __call__(self, update: int) -> None:
        """Publish step-numbered PNGs using native conditioning and pixel rounding."""
        config, model = (self.config, self.model)
        preview = config.preview
        if preview is None or config.model.vae is None or config.model.text_encoder is None:
            raise ValueError("FLUX previews require samples, vae and text assets")
        device = next(model.parameters()).device
        directory = Path(config.output.directory) / "previews"

        def load_captions() -> tuple[tuple[FluxTextFeatures, Caption | None], ...]:
            flux_text = load_flux_text(
                Path(config.model.text_encoder),
                config.model.asset_path("clip_text"),
                device=torch.device("cpu"),
                max_length=config.model.text.resolved_max_length("flux"),
                text_config=config.model.text,
            )
            captions = tuple((flux_text.encode([sample.prompt]), None) for sample in preview.samples)
            del flux_text
            return captions

        captions = self.cached_captions(load_captions)
        vae = load_flux_vae(Path(config.model.vae)).eval().requires_grad_(False)
        channels = model.config.out_channels // 4
        vae_channels = vae.config.latent_channels
        if vae_channels != channels:
            raise ValueError("Preview vae/model latent channels differ")

        def render(entry: tuple[SamplingConfig, tuple[FluxTextFeatures, Caption | None]]) -> tuple[torch.Tensor, ...]:
            sample, (positive, negative) = entry
            latents = generate_latents(
                model,
                positive,
                sample,
                allow_cpu_frozen=config.execution.offloads_frozen_weights,
                negative=negative,
                condition=build_flux_condition(vae, sample, variant=model.config.variant),
                vae_factor=vae.config.spatial_factor,
            )
            pixels = decode_on_device(vae, latents, device=device)[0]
            return (pixels,)

        publish_images(
            zip(preview.samples, captions, strict=True),
            render,
            directory=directory,
            update=update,
            rounding="nearest",
            save=save_image,
        )
