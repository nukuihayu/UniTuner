"""Own the native Qwen Image image generation stages and artifact validation."""

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
from unitunner.models.qwen_image.config.model import QwenImageConfig
from unitunner.models.qwen_image.data.media import read_inpainting_mask
from unitunner.models.qwen_image.inference.conditioning import build_qwen_image_references
from unitunner.models.qwen_image.loading import load_qwen_image, load_qwen_image_vae
from unitunner.models.qwen_image.projections import adapter_projections as qwen_adapter_projections
from unitunner.models.qwen_image.text.encoder import load_qwen_image_text


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
    inpainting_mask = None
    variant = QwenImageConfig.model_validate({"variant": config.model.variant}).variant
    vae = (
        load_qwen_image_vae(Path(vae_path), input_channels=4 if variant == "layered" else 3)
        .eval()
        .requires_grad_(False)
    )
    if vae.config.input_channels != (4 if variant == "layered" else 3):
        raise ValueError("Qwen vae RGB/RGBA channels differ from the requested variant")
    conditions = build_qwen_image_references(
        vae,
        tuple(Path(path) for path in config.sampling.reference_images),
        resize=config.sampling.conditioning_resize,
        resolution=(config.sampling.width, config.sampling.height),
    )
    qwen_text = load_qwen_image_text(
        Path(text_path),
        device=device,
        dtype=torch.float32,
        variant=variant,
        source_format="qwen" if config.model.text_source_format is None else config.model.text_source_format,
        text_config=config.model.text,
    )
    images = conditions.images if variant.startswith("edit") else None
    prompt = config.sampling.prompt
    if not prompt.strip() and config.sampling.caption_language is not None:
        prompt = qwen_text.describe_image(
            conditions.images[0], language=config.sampling.caption_language, seed=config.sampling.seed
        )
    positive = qwen_text.encode_prompt(prompt, references=images)
    positive = Caption(positive.features.to(torch.bfloat16).float())
    if config.sampling.guidance != 1:
        negative = qwen_text.encode_prompt(
            " " if config.sampling.negative_prompt is None else config.sampling.negative_prompt, references=images
        )
        negative = Caption(negative.features.to(torch.bfloat16).float())
    del qwen_text
    if config.sampling.mask_image is not None:
        inpainting_mask = read_inpainting_mask(
            Path(config.sampling.mask_image), (config.sampling.width, config.sampling.height)
        )
    references = conditions.latents
    model = load_qwen_image(
        Path(config.model.weights), variant=config.model.variant, device=device, dtype=torch.float32
    ).eval()
    channels = model.config.out_channels
    vae_channels = vae.config.latent_channels
    if model.config.variant != config.model.variant or channels != vae_channels:
        raise ValueError("Generation model variant/channels differ from the supplied model/vae")
    adapter = None
    targets = None
    with ExitStack() as stack:
        stack.enter_context(torch.autocast(device.type, enabled=False))
        if config.adapter is not None:
            targets = qwen_adapter_projections(model)
            adapter = LoadedAdapter(
                Path(config.adapter.weights),
                config.adapter.config,
                targets,
                external_prefixes=adapter_prefixes(targets, algorithm=config.adapter.config.algorithm),
            )
            stack.callback(adapter.close)
        latents = generate_latents(
            model,
            positive,
            config.sampling,
            negative=negative,
            references=references,
            inpainting_mask=inpainting_mask,
            vae_factor=vae.config.spatial_factor,
        ).cpu()
    if config.adapter is not None:
        del adapter, targets
    del model
    pixels = decode_on_device(vae, latents, device=device)
    return publish_images(pixels, Path(config.output), rounding="truncate")
