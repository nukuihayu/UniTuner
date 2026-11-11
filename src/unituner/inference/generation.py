"""Select a family-owned image generation application after validating the device."""

from typing import Protocol, cast

import torch

from unituner.config.generation import GenerationConfig
from unituner.execution.runtime import runtime_scope
from unituner.models.catalog import image_generation_provider
from unituner.models.providers import load_provider


class ImageGeneration(Protocol):
    """Describe the callable exported by each optional image-family application."""

    def __call__(self, config: GenerationConfig, *, device: torch.device) -> int:
        """Generate local output with explicit device ownership."""
        ...


@runtime_scope
def run(config: GenerationConfig, *, device: str) -> int:
    """Generate with native model stages; load providers only for the selected family."""
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("Generation entry supports CPU and CUDA devices")
    generate = cast(ImageGeneration, load_provider(image_generation_provider(config.model.family)))
    return generate(config, device=selected)
