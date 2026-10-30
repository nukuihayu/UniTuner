"""Load native image encoders with shared artifact options and lazy text imports."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from unituner.config.model import EncoderModelConfig
from unituner.models.catalog import ModelFamily
from unituner.models.flux.loading import load_flux_vae
from unituner.models.flux.vae import FluxVAE
from unituner.models.flux2.loading import load_flux2_vae
from unituner.models.flux2.vae import Flux2VAE
from unituner.models.qwen_image.loading import load_qwen_image_vae
from unituner.models.qwen_image.vae import QwenImageVAE
from unituner.models.qwen_image_2_1.loading import load_qwen_image_21_vae
from unituner.models.qwen_image_2_1.vae import QwenImage21VAE
from unituner.models.z_image.loading import load_z_image_vae
from unituner.models.z_image.vae import ZImageVAE

if TYPE_CHECKING:
    from unituner.models.flux.text.encoder import FluxTextEncoder
    from unituner.models.flux2.text.encoder import Flux2TextEncoder
    from unituner.models.flux3.text.encoder import Flux3TextEncoder
    from unituner.models.krea2.text.encoder import Krea2TextEncoder
    from unituner.models.moss_image.text.encoder import MossImageTextEncoder
    from unituner.models.qwen_image.text.encoder import QwenImageTextEncoder
    from unituner.models.qwen_image_2_1.text.encoder import QwenImage21TextEncoder
    from unituner.models.z_image.text.encoder import ZImageTextEncoder

    type PromptTextEncoder = (
        Flux2TextEncoder | Flux3TextEncoder | MossImageTextEncoder | QwenImageTextEncoder | ZImageTextEncoder
    )
    type ImageTextEncoder = PromptTextEncoder | FluxTextEncoder | Krea2TextEncoder | QwenImage21TextEncoder

type ImageVAE = FluxVAE | Flux2VAE | QwenImageVAE | QwenImage21VAE | ZImageVAE


def load_image_vae(
    path: Path, *, family: str, device: torch.device, input_channels: Literal[3, 4] | None = None
) -> ImageVAE:
    """Load a frozen FP32 VAE with the selected family's native geometry."""
    match family:
        case "z-image":
            vae = load_z_image_vae(path, device=device)
        case "flux":
            vae = load_flux_vae(path, device=device)
        case "flux2" | "flux3" | "moss-image":
            vae = load_flux2_vae(path, device=device)
        case "qwen-image" | "krea2":
            vae = load_qwen_image_vae(path, device=device, input_channels=input_channels)
        case "qwen-image-2-1":
            vae = load_qwen_image_21_vae(path, device=device)
        case _:
            raise ValueError(f"No image VAE for {family}")
    return vae.eval().requires_grad_(False)


def load_image_text(model: EncoderModelConfig, *, device: torch.device) -> ImageTextEncoder:
    """Load only the selected text provider, preserving native tokenizer and precision options."""
    if model.text_encoder is None:
        raise ValueError("Image text encoding requires model.text_encoder")
    path = Path(model.text_encoder)
    if model.family == ModelFamily.FLUX:
        from unituner.models.flux.text.encoder import load_flux_text

        return load_flux_text(
            path,
            model.asset_path("clip_text"),
            device=device,
            dtype=torch.float32,
            max_length=model.text.resolved_max_length("flux"),
            text_config=model.text,
        )
    if model.family == ModelFamily.FLUX2:
        if model.version not in ("dev", "klein-4b", "klein-9b"):
            raise ValueError("FLUX.2 requires model.version")
        from unituner.models.flux2.text.encoder import load_flux2_text

        return load_flux2_text(path, version=model.version, device=device, dtype=torch.float32, text_config=model.text)
    if model.family == ModelFamily.FLUX3:
        from unituner.models.flux3.text.encoder import load_flux3_text

        return load_flux3_text(path, device=device, dtype=torch.float32, text_config=model.text)
    if model.family == ModelFamily.MOSS_IMAGE:
        from unituner.models.moss_image.text.encoder import load_moss_image_text

        return load_moss_image_text(
            path,
            device=device,
            dtype=torch.float32,
            max_length=model.text.resolved_max_length("moss-image"),
            text_config=model.text,
        )
    if model.family == ModelFamily.KREA2:
        from unituner.models.krea2.text.encoder import load_krea2_text

        return load_krea2_text(
            path,
            device=device,
            dtype=torch.float32,
            config=model.text,
            source_format=model.text_source_format or "auto",
        )
    if model.family == ModelFamily.QWEN_IMAGE_2_1:
        from unituner.models.qwen_image_2_1.text.encoder import load_qwen_image_21_text

        return load_qwen_image_21_text(
            path,
            device=device,
            dtype=torch.float32,
            text_config=model.text,
            source_format=model.text_source_format or "auto",
        )
    if model.family == ModelFamily.QWEN_IMAGE:
        from unituner.models.qwen_image.config.model import QwenImageConfig
        from unituner.models.qwen_image.text.encoder import load_qwen_image_text

        variant = QwenImageConfig.model_validate({"variant": model.variant}).variant
        return load_qwen_image_text(
            path,
            device=device,
            dtype=torch.float32,
            variant=variant,
            text_config=model.text,
            source_format=model.text_source_format or "auto",
        )
    if model.family == ModelFamily.Z_IMAGE:
        from unituner.models.z_image.text.encoder import load_z_image_text

        return load_z_image_text(
            path,
            device=device,
            dtype=torch.float32,
            max_length=model.text.resolved_max_length("z-image"),
            text_config=model.text,
        )
    raise ValueError(f"No image text provider for {model.family}")
