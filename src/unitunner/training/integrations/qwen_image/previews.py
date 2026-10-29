"""Render Qwen previews from current reference bytes within the caller's preview context."""

from importlib import import_module
from pathlib import Path

import torch

from unitunner.config.sampling import SamplingConfig
from unitunner.config.training import TrainingConfig
from unitunner.data.media.image import save_image
from unitunner.inference.decoding import decode_on_device
from unitunner.inference.sampling import generate_latents
from unitunner.models.components.conditioning import Caption
from unitunner.models.qwen_image.data.media import read_inpainting_mask
from unitunner.models.qwen_image.inference.conditioning import build_qwen_image_references
from unitunner.models.qwen_image.loading import load_qwen_image_vae
from unitunner.models.qwen_image.transformer import QwenImageTransformer
from unitunner.training.previews.publication import publish_images


class ImagePreviews:
    """Borrow live Qwen Image resources and publish previews inside the caller-owned context."""

    def __init__(self, config: TrainingConfig, model: QwenImageTransformer) -> None:
        if config.model.family != "qwen-image":
            raise ValueError("Preview configuration and native model family differ")
        if config.model.variant != model.config.variant:
            raise ValueError("Preview configuration and artifact variant differ")
        self.config = config
        self.model = model

    def __call__(self, update: int) -> None:
        """Publish each RGB/RGBA layer while borrowing the active FP32 denoiser.

        The caller owns no-grad, evaluation mode and RNG restoration. Reload references
        and encode captions on each preview so edits to image or text assets take effect.
        Keep the VAE and text encoder on CPU until decode, and park the VAE even
        on failure. No denoiser parameter, adapter or residency installation changes.
        """
        config, model = (self.config, self.model)
        preview = config.preview
        if preview is None or config.model.vae is None or config.model.text_encoder is None:
            raise ValueError("Qwen previews require samples, vae and text assets")
        device = next(model.parameters()).device
        vae = (
            load_qwen_image_vae(Path(config.model.vae), input_channels=4 if model.config.variant == "layered" else 3)
            .eval()
            .requires_grad_(False)
        )
        if vae.config.latent_channels != model.config.out_channels:
            raise ValueError("Qwen preview vae/model latent channels differ")
        if vae.config.input_channels != (4 if model.config.variant == "layered" else 3):
            raise ValueError("Qwen preview vae RGB/RGBA channels differ from variant")
        provider = import_module("unitunner.models.qwen_image.text.encoder")
        text = provider.load_qwen_image_text(
            Path(config.model.text_encoder),
            device=torch.device("cpu"),
            dtype=torch.float32,
            variant=model.config.variant,
            source_format=config.model.text_source_format or "auto",
            text_config=config.model.text,
        )
        directory = Path(config.output.directory) / "previews"

        def render(sample: SamplingConfig) -> tuple[torch.Tensor, ...]:
            conditions = build_qwen_image_references(
                vae,
                tuple(Path(path) for path in sample.reference_images),
                resize=sample.conditioning_resize,
                resolution=(sample.width, sample.height),
            )
            images = conditions.images if model.config.variant.startswith("edit") else None
            prompt = sample.prompt
            if not prompt.strip() and sample.caption_language is not None:
                prompt = text.describe_image(conditions.images[0], language=sample.caption_language, seed=sample.seed)
            positive = text.encode_prompt(prompt, references=images)
            positive = Caption(positive.features.to(torch.bfloat16).float())
            negative = None
            if sample.guidance != 1:
                negative = text.encode_prompt(
                    " " if sample.negative_prompt is None else sample.negative_prompt, references=images
                )
                negative = Caption(negative.features.to(torch.bfloat16).float())
            mask = (
                None
                if sample.mask_image is None
                else read_inpainting_mask(Path(sample.mask_image), (sample.width, sample.height))
            )
            latents = generate_latents(
                model,
                positive,
                sample,
                allow_cpu_frozen=config.execution.offloads_frozen_weights,
                negative=negative,
                references=conditions.latents,
                inpainting_mask=mask,
                vae_factor=vae.config.spatial_factor,
            )
            pixels = decode_on_device(vae, latents, device=device)
            return tuple(pixels)

        publish_images(
            preview.samples, render, directory=directory, update=update, rounding="truncate", save=save_image
        )
