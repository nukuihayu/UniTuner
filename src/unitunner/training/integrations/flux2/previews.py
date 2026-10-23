"""Publish native FLUX.2 previews without owning training state."""

from pathlib import Path

import torch

from unitunner.config.sampling import SamplingConfig
from unitunner.config.training import TrainingConfig
from unitunner.data.media.image import save_image
from unitunner.inference.decoding import decode_on_device
from unitunner.inference.sampling import generate_latents
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux2.inference.conditioning import build_flux2_references
from unitunner.models.flux2.loading import load_flux2_vae
from unitunner.models.flux2.text.encoder import load_flux2_text
from unitunner.models.flux2.transformer import Flux2Transformer
from unitunner.training.previews.publication import CaptionCache, publish_images


class ImagePreviews(CaptionCache[tuple[tuple[Caption, Caption | None], ...]]):
    """Borrow the live FLUX.2 model and cache CPU text features between previews.

    Construction performs no IO. The caller owns eval/no-grad/RNG restoration;
    each call parks its VAE on CPU even when generation or publication fails.
    """

    def __init__(self, config: TrainingConfig, model: Flux2Transformer) -> None:
        if config.model.family != "flux2":
            raise ValueError("Preview configuration and native model family differ")
        if config.model.variant != model.config.variant:
            raise ValueError("Preview configuration and artifact variant differ")
        if config.model.version != model.config.version:
            raise ValueError("Preview configuration and artifact version differ")
        self.config = config
        self.model = model
        self.captions: tuple[tuple[Caption, Caption | None], ...] | None = None

    def __call__(self, update: int) -> None:
        """Publish step-numbered PNGs using native conditioning and pixel rounding."""
        config, model = (self.config, self.model)
        preview = config.preview
        if preview is None or config.model.vae is None or config.model.text_encoder is None:
            raise ValueError("FLUX.2 previews require samples, vae and text assets")
        device = next(model.parameters()).device
        directory = Path(config.output.directory) / "previews"
        text_path = Path(config.model.text_encoder)

        def load_captions() -> tuple[tuple[Caption, Caption | None], ...]:
            text = load_flux2_text(
                text_path,
                version=model.config.version,
                device=torch.device("cpu"),
                dtype=torch.float32,
                text_config=config.model.text,
            )
            captions = tuple(
                (
                    text.encode_prompt(sample.prompt),
                    text.encode_prompt(" " if sample.negative_prompt is None else sample.negative_prompt)
                    if model.config.variant == "base" or sample.negative_prompt is not None
                    else None,
                )
                for sample in preview.samples
            )
            del text
            return captions

        captions = self.cached_captions(load_captions)
        vae = load_flux2_vae(Path(config.model.vae)).eval().requires_grad_(False)
        channels = model.config.in_channels
        vae_channels = 4 * vae.config.latent_channels
        if vae_channels != channels:
            raise ValueError("Preview vae/model latent channels differ")

        def render(entry: tuple[SamplingConfig, tuple[Caption, Caption | None]]) -> tuple[torch.Tensor, ...]:
            sample, (positive, negative) = entry
            latents = generate_latents(
                model,
                positive,
                sample,
                allow_cpu_frozen=config.execution.offloads_frozen_weights,
                negative=negative,
                references=build_flux2_references(
                    vae,
                    tuple(Path(path) for path in sample.reference_images),
                    original=sample.conditioning_resize == "original",
                ),
                vae_factor=vae.config.spatial_factor,
            )
            pixels = decode_on_device(vae, latents, device=device)[0]
            return (pixels,)

        publish_images(
            zip(preview.samples, captions, strict=True),
            render,
            directory=directory,
            update=update,
            rounding="truncate",
            save=save_image,
        )
