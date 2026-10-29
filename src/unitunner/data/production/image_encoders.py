"""Own image-cache VAE/text loading and preserve native encoder options."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from unitunner.config.model import EncoderModelConfig
from unitunner.models.catalog import ModelFamily
from unitunner.models.flux2.loading import load_flux2_vae
from unitunner.models.qwen_image.loading import load_qwen_image_vae
from unitunner.progress import status

if TYPE_CHECKING:
    from unitunner.models.flux.text.encoder import FluxTextEncoder
    from unitunner.models.flux.vae import FluxVAE
    from unitunner.models.flux2.text.encoder import Flux2TextEncoder
    from unitunner.models.flux2.vae import Flux2VAE
    from unitunner.models.flux3.text.encoder import Flux3TextEncoder
    from unitunner.models.krea2.text.encoder import Krea2TextEncoder
    from unitunner.models.moss_image.text.encoder import MossImageTextEncoder
    from unitunner.models.qwen_image.text.encoder import QwenImageTextEncoder
    from unitunner.models.qwen_image.vae import QwenImageVAE
    from unitunner.models.qwen_image_2_1.text.encoder import QwenImage21TextEncoder
    from unitunner.models.qwen_image_2_1.vae import QwenImage21VAE
    from unitunner.models.z_image.text.encoder import ZImageTextEncoder
    from unitunner.models.z_image.vae import ZImageVAE


def load_image_encoders(
    model: EncoderModelConfig, *, device: torch.device
) -> tuple[
    ZImageVAE | FluxVAE | Flux2VAE | QwenImageVAE | QwenImage21VAE,
    ZImageTextEncoder
    | FluxTextEncoder
    | Flux2TextEncoder
    | Flux3TextEncoder
    | MossImageTextEncoder
    | Krea2TextEncoder
    | QwenImageTextEncoder
    | QwenImage21TextEncoder,
]:
    """Load frozen FP32 cache VAEs and independently configured native text providers."""
    vae_path, text_path = model.vae, model.text_encoder
    if vae_path is None or text_path is None:
        raise ValueError("Cache production requires model.vae and model.text_encoder")
    secondary = model.asset_path("clip_text") if model.family == ModelFamily.FLUX else None
    if model.family == ModelFamily.KREA2:
        from unitunner.models.krea2.text.encoder import load_krea2_text

        status("[cache] Loading qwen image vae")
        vae = load_qwen_image_vae(Path(vae_path), device=device).eval().requires_grad_(False)
        status("[cache] Loaded qwen image vae")
        status("[cache] Loading krea2 text")
        text = load_krea2_text(
            Path(text_path),
            device=device,
            dtype=torch.float32,
            config=model.text,
            source_format=model.text_source_format or "auto",
        )
        status("[cache] Loaded krea2 text")

    elif model.family == ModelFamily.QWEN_IMAGE_2_1:
        from unitunner.models.qwen_image_2_1.loading import load_qwen_image_21_vae
        from unitunner.models.qwen_image_2_1.text.encoder import load_qwen_image_21_text

        status("[cache] Loading qwen image 21 vae")
        vae = load_qwen_image_21_vae(Path(vae_path), device=device).eval().requires_grad_(False)
        status("[cache] Loaded qwen image 21 vae")
        status("[cache] Loading qwen image 21 text")
        text = load_qwen_image_21_text(
            Path(text_path),
            device=device,
            dtype=torch.float32,
            source_format=model.text_source_format or "auto",
            text_config=model.text,
        )
        status("[cache] Loaded qwen image 21 text")

    elif model.family == ModelFamily.QWEN_IMAGE:
        from unitunner.models.qwen_image.config.model import QwenImageConfig
        from unitunner.models.qwen_image.text.encoder import load_qwen_image_text

        variant = QwenImageConfig.model_validate({"variant": model.variant}).variant
        status("[cache] Loading qwen image vae")
        vae = (
            load_qwen_image_vae(Path(vae_path), device=device, input_channels=4 if variant == "layered" else 3)
            .eval()
            .requires_grad_(False)
        )
        status("[cache] Loaded qwen image vae")
        status("[cache] Loading qwen image text")
        text = load_qwen_image_text(
            Path(text_path),
            device=device,
            variant=variant,
            dtype=torch.float32,
            source_format=model.text_source_format or "auto",
            text_config=model.text,
        )
        status("[cache] Loaded qwen image text")

    elif model.family == ModelFamily.FLUX:
        from unitunner.models.flux.loading import load_flux_vae
        from unitunner.models.flux.text.encoder import load_flux_text

        if secondary is None:
            raise ValueError("Flux requires a local CLIP text directory")
        status("[cache] Loading flux vae")
        vae = load_flux_vae(Path(vae_path), device=device).eval().requires_grad_(False)
        status("[cache] Loaded flux vae")
        status("[cache] Loading flux text")
        text = load_flux_text(
            Path(text_path),
            Path(secondary),
            device=device,
            dtype=torch.float32,
            max_length=model.text.resolved_max_length("flux"),
            text_config=model.text,
        )
        status("[cache] Loaded flux text")
    elif model.family == ModelFamily.FLUX2:
        from unitunner.models.flux2.text.encoder import load_flux2_text

        if model.version not in ("dev", "klein-4b", "klein-9b"):
            raise ValueError("FLUX.2 requires model.version")
        status("[cache] Loading flux2 vae")
        vae = load_flux2_vae(Path(vae_path), device=device).eval().requires_grad_(False)
        status("[cache] Loaded flux2 vae")
        status("[cache] Loading flux2 text")
        text = load_flux2_text(
            Path(text_path),
            version=model.version,
            device=device,
            dtype=torch.float32,
            text_config=model.text,
        )
        status("[cache] Loaded flux2 text")
    elif model.family == ModelFamily.FLUX3:
        from unitunner.models.flux3.text.encoder import load_flux3_text

        status("[cache] Loading flux2 vae")
        vae = load_flux2_vae(Path(vae_path), device=device).eval().requires_grad_(False)
        status("[cache] Loaded flux2 vae")
        status("[cache] Loading flux3 text")
        text = load_flux3_text(Path(text_path), device=device, dtype=torch.float32, text_config=model.text)
        status("[cache] Loaded flux3 text")
    elif model.family == ModelFamily.MOSS_IMAGE:
        from unitunner.models.moss_image.text.encoder import load_moss_image_text

        status("[cache] Loading flux2 vae")
        vae = load_flux2_vae(Path(vae_path), device=device).eval().requires_grad_(False)
        status("[cache] Loaded flux2 vae")
        status("[cache] Loading moss image text")
        text = load_moss_image_text(
            Path(text_path),
            device=device,
            dtype=torch.float32,
            max_length=model.text.resolved_max_length("moss-image"),
            text_config=model.text,
        )
        status("[cache] Loaded moss image text")
    elif model.family == ModelFamily.Z_IMAGE:
        from unitunner.models.z_image.loading import load_z_image_vae
        from unitunner.models.z_image.text.encoder import load_z_image_text

        status("[cache] Loading z image vae")
        vae = load_z_image_vae(Path(vae_path), device=device).eval().requires_grad_(False)
        status("[cache] Loaded z image vae")
        status("[cache] Loading z image text")
        text = load_z_image_text(
            Path(text_path),
            device=device,
            dtype=torch.float32,
            max_length=model.text.resolved_max_length("z-image"),
            text_config=model.text,
        )
        status("[cache] Loaded z image text")
    else:
        raise ValueError(f"No image cache encoders for {model.family}")
    return vae, text
