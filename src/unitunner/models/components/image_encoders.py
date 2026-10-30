"""Load native image encoders with shared artifact options and lazy text imports."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from unitunner.config.model import EncoderModelConfig
from unitunner.models.catalog import ModelFamily
from unitunner.models.flux.loading import load_flux_vae
from unitunner.models.flux.vae import FluxVAE
from unitunner.models.flux2.loading import load_flux2_vae
from unitunner.models.flux2.vae import Flux2VAE
from unitunner.models.qwen_image.loading import load_qwen_image_vae
from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.models.qwen_image_2_1.loading import load_qwen_image_21_vae
from unitunner.models.qwen_image_2_1.vae import QwenImage21VAE
from unitunner.models.z_image.loading import load_z_image_vae
from unitunner.models.z_image.vae import ZImageVAE

if TYPE_CHECKING:
    from unitunner.models.flux.text.encoder import FluxTextEncoder
    from unitunner.models.flux2.text.encoder import Flux2TextEncoder
    from unitunner.models.flux3.text.encoder import Flux3TextEncoder
    from unitunner.models.krea2.text.encoder import Krea2TextEncoder
    from unitunner.models.moss_image.text.encoder import MossImageTextEncoder
    from unitunner.models.qwen_image.text.encoder import QwenImageTextEncoder
    from unitunner.models.qwen_image_2_1.text.encoder import QwenImage21TextEncoder
    from unitunner.models.z_image.text.encoder import ZImageTextEncoder

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
        from unitunner.models.flux.text.encoder import load_flux_text

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
        from unitunner.models.flux2.text.encoder import load_flux2_text

        return load_flux2_text(path, version=model.version, device=device, dtype=torch.float32, text_config=model.text)
    if model.family == ModelFamily.FLUX3:
        from unitunner.models.flux3.text.encoder import load_flux3_text

        return load_flux3_text(path, device=device, dtype=torch.float32, text_config=model.text)
    if model.family == ModelFamily.MOSS_IMAGE:
        from unitunner.models.moss_image.text.encoder import load_moss_image_text

        return load_moss_image_text(
            path,
            device=device,
            dtype=torch.float32,
            max_length=model.text.resolved_max_length("moss-image"),
            text_config=model.text,
        )
    if model.family == ModelFamily.KREA2:
        from unitunner.models.krea2.text.encoder import load_krea2_text

        return load_krea2_text(
            path,
            device=device,
            dtype=torch.float32,
            config=model.text,
            source_format=model.text_source_format or "auto",
        )
    if model.family == ModelFamily.QWEN_IMAGE_2_1:
        from unitunner.models.qwen_image_2_1.text.encoder import load_qwen_image_21_text

        return load_qwen_image_21_text(
            path,
            device=device,
            dtype=torch.float32,
            text_config=model.text,
            source_format=model.text_source_format or "auto",
        )
    if model.family == ModelFamily.QWEN_IMAGE:
        from unitunner.models.qwen_image.config.model import QwenImageConfig
        from unitunner.models.qwen_image.text.encoder import load_qwen_image_text

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
        from unitunner.models.z_image.text.encoder import load_z_image_text

        return load_z_image_text(
            path,
            device=device,
            dtype=torch.float32,
            max_length=model.text.resolved_max_length("z-image"),
            text_config=model.text,
        )
    raise ValueError(f"No image text provider for {model.family}")
