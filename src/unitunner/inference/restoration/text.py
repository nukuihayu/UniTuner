"""Encode restoration prompts using only the selected family's native text provider."""

from pathlib import Path

import torch

from unitunner.config.restoration_inference import RestorationInferenceConfig
from unitunner.models.catalog import ModelFamily
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux.conditioning import FluxTextFeatures
from unitunner.models.krea2.conditioning import Krea2TextFeatures, batch_text


def encode_caption(
    config: RestorationInferenceConfig, *, device: torch.device
) -> Caption | Krea2TextFeatures | FluxTextFeatures:
    """Keep native feature layout, source format and precision at the text boundary."""
    text_path = config.model.text_encoder
    if text_path is None:
        raise ValueError("Restoration requires model.text_encoder")
    caption: Caption | Krea2TextFeatures | FluxTextFeatures
    with torch.autocast(device.type, enabled=False):
        if config.model.family == ModelFamily.FLUX:
            from unitunner.models.flux.text.encoder import load_flux_text

            flux_text = load_flux_text(
                Path(text_path),
                config.model.asset_path("clip_text"),
                text_config=config.model.text,
                device=device,
                max_length=config.model.text.resolved_max_length("flux"),
            )
            caption = flux_text.encode([config.prompt])
            del flux_text
        elif config.model.family == ModelFamily.FLUX2:
            if config.model.version not in ("dev", "klein-4b", "klein-9b"):
                raise ValueError("FLUX.2 restoration requires model.version")
            from unitunner.models.flux2.text.encoder import load_flux2_text

            flux2_text = load_flux2_text(
                Path(text_path),
                text_config=config.model.text,
                version=config.model.version,
                device=device,
                dtype=torch.float32,
            )
            caption = flux2_text.encode_prompt(config.prompt)
            del flux2_text
        elif config.model.family == ModelFamily.FLUX3:
            from unitunner.models.flux3.text.encoder import load_flux3_text

            flux3_text = load_flux3_text(
                Path(text_path), text_config=config.model.text, device=device, dtype=torch.float32
            )
            caption = flux3_text.encode_prompt(config.prompt)
            del flux3_text
        elif config.model.family == ModelFamily.MOSS_IMAGE:
            from unitunner.models.moss_image.text.encoder import load_moss_image_text

            moss_text = load_moss_image_text(
                Path(text_path),
                text_config=config.model.text,
                device=device,
                dtype=torch.float32,
                max_length=config.model.text.resolved_max_length("moss-image"),
            )
            caption = moss_text.encode_prompt(config.prompt)
            del moss_text
        elif config.model.family == ModelFamily.KREA2:
            from unitunner.models.krea2.text.encoder import load_krea2_text

            krea_text = load_krea2_text(
                Path(text_path),
                device=device,
                dtype=torch.float32,
                config=config.model.text,
                source_format="transformers"
                if config.model.text_source_format is None
                else config.model.text_source_format,
            )
            caption = batch_text(krea_text.encode([config.prompt]))
            del krea_text
        elif config.model.family == ModelFamily.QWEN_IMAGE_2_1:
            from unitunner.models.qwen_image_2_1.text.encoder import load_qwen_image_21_text

            text21 = load_qwen_image_21_text(
                Path(text_path),
                text_config=config.model.text,
                device=device,
                dtype=torch.float32,
                source_format="transformers"
                if config.model.text_source_format is None
                else config.model.text_source_format,
            )
            caption = Caption(text21.encode([config.prompt])[0].features[None])
            del text21
        elif config.model.family == ModelFamily.QWEN_IMAGE:
            from unitunner.models.qwen_image.text.encoder import load_qwen_image_text

            text = load_qwen_image_text(
                Path(text_path),
                text_config=config.model.text,
                device=device,
                variant="original",
                dtype=torch.float32,
                source_format=config.model.text_source_format or "qwen",
            )
            caption = text.encode_prompt(config.prompt)
        elif config.model.family == ModelFamily.Z_IMAGE:
            from unitunner.models.z_image.text.encoder import load_z_image_text

            text = load_z_image_text(
                Path(text_path),
                text_config=config.model.text,
                device=device,
                dtype=torch.float32,
                max_length=config.model.text.resolved_max_length("z-image"),
            )
            caption = text.encode_prompt(config.prompt)
        else:
            raise ValueError(f"No restoration text provider for {config.model.family}")
    return caption
