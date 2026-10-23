"""Publish native Z-Image previews without owning training state."""

from pathlib import Path

import torch

from unitunner.config.sampling import SamplingConfig
from unitunner.config.training import TrainingConfig
from unitunner.data.media.image import save_image
from unitunner.inference.decoding import decode_on_device
from unitunner.inference.sampling import generate_latents
from unitunner.models.components.conditioning import Caption
from unitunner.models.z_image.loading import load_z_image_vae
from unitunner.models.z_image.text.encoder import load_z_image_text
from unitunner.models.z_image.transformer import ZImageTransformer
from unitunner.training.previews.publication import CaptionCache, publish_images


class ImagePreviews(CaptionCache[tuple[tuple[Caption, Caption | None], ...]]):
    """Borrow the live Z-Image model and cache CPU text features between previews.

    Construction performs no IO. The caller owns eval/no-grad/RNG restoration;
    each call parks its VAE on CPU even when generation or publication fails.
    """

    def __init__(self, config: TrainingConfig, model: ZImageTransformer) -> None:
        if config.model.family != "z-image":
            raise ValueError("Preview configuration and native model family differ")
        if config.model.variant != model.config.variant:
            raise ValueError("Preview configuration and artifact variant differ")
        self.config = config
        self.model = model
        self.captions: tuple[tuple[Caption, Caption | None], ...] | None = None

    def __call__(self, update: int) -> None:
        """Publish step-numbered PNGs using native conditioning and pixel rounding."""
        config, model = (self.config, self.model)
        preview = config.preview
        if preview is None or config.model.vae is None or config.model.text_encoder is None:
            raise ValueError("Z-Image previews require samples, vae and text assets")
        device = next(model.parameters()).device
        directory = Path(config.output.directory) / "previews"

        def load_captions() -> tuple[tuple[Caption, Caption | None], ...]:
            text = load_z_image_text(
                Path(config.model.text_encoder),
                device=torch.device("cpu"),
                dtype=torch.float32,
                max_length=config.model.text.resolved_max_length("z-image"),
                text_config=config.model.text,
            )
            captions = tuple(
                (
                    text.encode_prompt(sample.prompt),
                    None if sample.negative_prompt is None else text.encode_prompt(sample.negative_prompt),
                )
                for sample in preview.samples
            )
            del text
            return captions

        captions = self.cached_captions(load_captions)
        vae = load_z_image_vae(Path(config.model.vae)).eval().requires_grad_(False)
        channels = model.config.in_channels
        vae_channels = vae.config.latent_channels
        if vae_channels != channels:
            raise ValueError("Preview vae/model latent channels differ")

        def render(entry: tuple[SamplingConfig, tuple[Caption, Caption | None]]) -> tuple[torch.Tensor, ...]:
            sample, (positive, negative) = entry
            latents = generate_latents(
                model, positive, sample, negative=negative, vae_factor=2 ** (len(vae.config.block_channels) - 1)
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
