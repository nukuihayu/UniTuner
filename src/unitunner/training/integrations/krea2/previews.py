"""Publish Krea2 previews while borrowing the live model and adapter parameter identities."""

from pathlib import Path

import torch

from unitunner.config.sampling import SamplingConfig
from unitunner.config.training import TrainingConfig
from unitunner.data.media.image import save_image
from unitunner.inference.decoding import decode_on_device
from unitunner.inference.sampling import generate_latents
from unitunner.models.krea2.conditioning import Krea2TextFeatures
from unitunner.models.krea2.inference.conditioning import encode_krea2_prompts
from unitunner.models.krea2.text.encoder import load_krea2_text
from unitunner.models.krea2.transformer import Krea2Transformer
from unitunner.models.qwen_image.loading import load_qwen_image_vae
from unitunner.training.previews.publication import publish_images


class ImagePreviews:
    """Borrow live Krea2 resources and publish previews inside the caller-owned context."""

    def __init__(self, config: TrainingConfig, model: Krea2Transformer) -> None:
        if config.model.family != "krea2":
            raise ValueError("Preview configuration and native model family differ")
        if config.model.variant != model.config.variant:
            raise ValueError("Preview configuration and artifact variant differ")
        self.config = config
        self.model = model

    def __call__(self, update: int) -> None:
        """Encode CPU captions and decode live Krea2 samples to truncated RGB PNGs.

        The shared preview context owns no-grad, eval and RNG restoration. This
        function borrows active adapters, moves only the VAE for decode and returns
        it to CPU even on failure. No optimizer/model/adapter storage is replaced.
        """
        config, model = (self.config, self.model)
        preview = config.preview
        if preview is None or config.model.vae is None or config.model.text_encoder is None:
            raise ValueError("Krea2 previews require vae/text assets and samples")
        vae = load_qwen_image_vae(Path(config.model.vae)).eval().requires_grad_(False)
        if vae.config.input_channels != 3 or vae.config.latent_channels != 16 or vae.config.spatial_factor != 8:
            raise ValueError("Krea2 previews require the shared RGB latent16/factor8 vae")
        text = load_krea2_text(
            Path(config.model.text_encoder),
            device=torch.device("cpu"),
            dtype=torch.float32,
            config=config.model.text,
            source_format=config.model.text_source_format or "auto",
        )
        captions = tuple(encode_krea2_prompts(text, sample) for sample in preview.samples)
        del text
        directory = Path(config.output.directory) / "previews"
        device = next(model.parameters()).device

        def render(
            entry: tuple[SamplingConfig, tuple[Krea2TextFeatures, Krea2TextFeatures | None]],
        ) -> tuple[torch.Tensor, ...]:
            sample, (positive, negative) = entry
            latents = generate_latents(model, positive, sample, negative=negative)
            pixels = decode_on_device(vae, latents.unsqueeze(2), device=device)
            return (pixels[0],)

        publish_images(
            zip(preview.samples, captions, strict=True),
            render,
            directory=directory,
            update=update,
            rounding="truncate",
            save=save_image,
        )
