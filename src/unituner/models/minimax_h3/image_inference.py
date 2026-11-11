"""Local text-to-image and reference-image editing with native H3-Image components."""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from unituner.models.minimax_h3.data.media.inputs import prepare_pixels, resize_frames, round_to_canvas_multiple
from unituner.models.minimax_h3.image import MiniMaxH3ImageTransformer, sample_image
from unituner.models.minimax_h3.image_vae import MiniMaxH3ImageVAE
from unituner.models.minimax_h3.text.encoder import MiniMaxH3TextEncoder
from unituner.models.minimax_h3.text.presentation import (
    MiniMaxH3TextVisual,
    build_image_presentation,
)


def official_size(width: int, height: int) -> tuple[int, int]:
    """Scale to the official example's 4096-square area on the 32-pixel grid."""
    if width <= 0 or height <= 0 or not 1 / 4 <= width / height <= 4:
        raise ValueError("H3-Image official aspect ratio must have positive sides within 1:4 to 4:1")
    area = 4096**2
    return (
        round_to_canvas_multiple(math.sqrt(area * width / height)),
        round_to_canvas_multiple(math.sqrt(area * height / width)),
    )


def _resize_reference(source: np.ndarray, min_short_edge: int, max_short_edge: int) -> Tensor:
    """Match MiniMaxH3ImageConditioning's short-edge clamp and 32-pixel grid."""
    height, width = source.shape[:2]
    scale = min(max(min(height, width), min_short_edge), max_short_edge) / min(height, width)
    size = (round_to_canvas_multiple(width * scale), round_to_canvas_multiple(height * scale))
    return resize_frames((source,), size)


@torch.no_grad()
def encode_image_condition(
    vae: MiniMaxH3ImageVAE,
    text_encoder: MiniMaxH3TextEncoder,
    prompt: str,
    *,
    width: int | None = None,
    height: int | None = None,
    size_policy: Literal["comfyui", "official"] = "comfyui",
    images: Sequence[Path] = (),
    model_device: torch.device,
) -> tuple[Tensor, Tensor, tuple[Tensor, ...], tuple[int, int]]:
    """Encode the same ordered H3-Image text and reference payload for sampling or training."""
    vae_device = next(vae.parameters()).device
    if len(images) > 16:
        raise ValueError("MiniMax-H3-Image supports at most 16 reference images")
    references = []
    visuals = []
    reference_size = None
    for path in images:
        with Image.open(path) as image, image.convert("RGB") as rgb:
            source = np.asarray(rgb)
        vision = _resize_reference(source, 1024, 1024)
        vae_image = _resize_reference(source, 1024, 2048)
        if reference_size is None:
            reference_size = (vae_image.shape[2], vae_image.shape[1])
        pixels = prepare_pixels(vae_image).to(vae_device)
        references.append(vae.encode(pixels).to(model_device))
        visuals.append(MiniMaxH3TextVisual(vision))
    if size_policy not in ("comfyui", "official"):
        raise ValueError("H3-Image size_policy must be comfyui or official")
    default_size = reference_size or (2048, 2048)
    if size_policy == "official" and (width is None or height is None):
        default_size = official_size(*default_size)
    width = width if width is not None else default_size[0]
    height = height if height is not None else default_size[1]
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("H3-Image output dimensions must be positive multiples of 32")
    presentation = build_image_presentation(prompt, visuals)
    encoded = text_encoder.encode(presentation)
    return (
        encoded.hidden_states.unsqueeze(0).to(model_device),
        encoded.token_tags.unsqueeze(0).to(model_device),
        tuple(references),
        (width, height),
    )


@torch.no_grad()
def generate_image(
    model: MiniMaxH3ImageTransformer,
    vae: MiniMaxH3ImageVAE,
    text_encoder: MiniMaxH3TextEncoder,
    prompt: str,
    *,
    width: int | None = None,
    height: int | None = None,
    size_policy: Literal["comfyui", "official"] = "comfyui",
    seed: int = 0,
    images: Sequence[Path] = (),
) -> Tensor:
    """Generate B3x1xHxW pixels in [-1,1], with ordered reference images for edits."""
    text, tags, references, (width, height) = encode_image_condition(
        vae,
        text_encoder,
        prompt,
        width=width,
        height=height,
        size_policy=size_policy,
        images=images,
        model_device=model.device,
    )
    noise = torch.randn(
        (1, 24, 1, height // 16, width // 16),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
        dtype=torch.float32,
    ).to(model.device)
    latents = sample_image(
        model,
        noise,
        text,
        tags,
        references=references,
        reference_seed=seed,
    )
    return vae.decode(latents.to(next(vae.parameters()).device))
