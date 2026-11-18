"""Encode restoration prompts using the selected family's native feature layout."""

from typing import TYPE_CHECKING, cast

import torch

from unituner.config.restoration_inference import RestorationInferenceConfig
from unituner.models.catalog import ModelFamily
from unituner.models.components.conditioning import Caption
from unituner.models.components.image_encoders import load_image_text
from unituner.models.flux.conditioning import FluxTextFeatures
from unituner.models.krea2.conditioning import Krea2TextFeatures, batch_text

if TYPE_CHECKING:
    from unituner.models.components.image_encoders import PromptTextEncoder
    from unituner.models.flux.text.encoder import FluxTextEncoder
    from unituner.models.krea2.text.encoder import Krea2TextEncoder
    from unituner.models.qwen_image_2_1.text.encoder import QwenImage21TextEncoder


def encode_caption(
    config: RestorationInferenceConfig, *, device: torch.device
) -> Caption | Krea2TextFeatures | FluxTextFeatures:
    """Keep native feature layout and disable autocast while loading/encoding text."""
    with torch.autocast(device.type, enabled=False):
        text = load_image_text(config.model, device=device)
        if config.model.family == ModelFamily.FLUX:
            return cast("FluxTextEncoder", text).encode([config.prompt])
        if config.model.family == ModelFamily.KREA2:
            return batch_text(cast("Krea2TextEncoder", text).encode([config.prompt]))
        if config.model.family == ModelFamily.QWEN_IMAGE_2_1:
            return Caption(cast("QwenImage21TextEncoder", text).encode([config.prompt])[0].features[None])
        return cast("PromptTextEncoder", text).encode_prompt(config.prompt)
