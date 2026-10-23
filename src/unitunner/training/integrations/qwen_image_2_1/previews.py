"""Publish Qwen2.1 previews from active model/adapter parameters inside the shared preview context."""

from pathlib import Path

import torch

from unitunner.config.sampling import SamplingConfig
from unitunner.config.training import TrainingConfig
from unitunner.data.media.image import save_image
from unitunner.inference.decoding import decode_on_device
from unitunner.inference.sampling import generate_latents
from unitunner.models.qwen_image_2_1.inference.conditioning import build_qwen_image_21_conditions
from unitunner.models.qwen_image_2_1.loading import load_qwen_image_21_vae
from unitunner.models.qwen_image_2_1.text.encoder import load_qwen_image_21_text
from unitunner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unitunner.training.previews.publication import publish_images


class ImagePreviews:
    """Borrow live Qwen Image 2.1 resources and publish previews inside the caller-owned context."""

    def __init__(self, config: TrainingConfig, model: QwenImage21Transformer) -> None:
        if config.model.family != "qwen-image-2-1":
            raise ValueError("Preview configuration and native model family differ")
        self.config = config
        self.model = model

    def __call__(self, update: int) -> None:
        """Encode current reference/text assets on CPU, borrow the denoiser and publish RGB/RGBA PNGs.

        The caller owns eval/no-grad/RNG restoration. No adapter/optimizer/model
        identity changes occur. VAE moves only for decode and returns to CPU even
        on failure; references and text retain independent insertion metadata.
        """
        config, model = (self.config, self.model)
        preview = config.preview
        if preview is None or config.model.vae is None or config.model.text_encoder is None:
            raise ValueError("Qwen Image2.1 previews require vae/text assets and samples")
        vae = load_qwen_image_21_vae(Path(config.model.vae)).eval().requires_grad_(False)
        if (
            vae.config.latent_channels != model.config.in_channels
            or model.config.out_channels != vae.config.latent_channels
        ):
            raise ValueError("Qwen Image2.1 preview model/vae channels differ")
        text = load_qwen_image_21_text(
            Path(config.model.text_encoder),
            device=torch.device("cpu"),
            dtype=torch.float32,
            source_format="transformers"
            if config.model.text_source_format is None
            else config.model.text_source_format,
            text_config=config.model.text,
        )
        directory = Path(config.output.directory) / "previews"
        device = next(model.parameters()).device

        def render(sample: SamplingConfig) -> tuple[torch.Tensor, ...]:
            conditions = build_qwen_image_21_conditions(vae, text, sample)
            latents = generate_latents(
                model,
                conditions.positive,
                sample,
                negative=conditions.negative,
                references=conditions.references,
                inpainting_mask=conditions.mask,
                vae_factor=vae.config.spatial_factor,
            )
            pixels = decode_on_device(vae, latents, device=device)
            return (pixels[0],)

        publish_images(
            preview.samples, render, directory=directory, update=update, rounding="truncate", save=save_image
        )
