"""Select a family-owned image generation application after validating the device."""

from importlib import import_module
from typing import Protocol, cast

import torch

from unitunner.config.generation import GenerationConfig
from unitunner.execution.runtime import runtime_scope
from unitunner.models.catalog import ModelFamily


class ImageGeneration(Protocol):
    """Describe the callable exported by each optional image-family application."""

    def __call__(self, config: GenerationConfig, *, device: torch.device) -> int:
        """Generate local output with explicit device ownership."""
        ...


@runtime_scope
def run(config: GenerationConfig, *, device: str) -> int:
    """Generate with native model stages; load optional dependencies only for the selected family."""
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("Generation entry supports CPU and CUDA devices")
    module = {
        ModelFamily.FLUX: "unitunner.inference.images",
        ModelFamily.FLUX2: "unitunner.inference.images",
        ModelFamily.FLUX3: "unitunner.inference.images",
        ModelFamily.MOSS_IMAGE: "unitunner.inference.images",
        ModelFamily.Z_IMAGE: "unitunner.inference.images",
        ModelFamily.QWEN_IMAGE: "unitunner.models.qwen_image.inference.run",
        ModelFamily.QWEN_IMAGE_2_1: "unitunner.models.qwen_image_2_1.inference.run",
        ModelFamily.KREA2: "unitunner.models.krea2.inference.run",
    }.get(config.model.family)
    if module is None:
        raise ValueError("Image generation requires a supported image model family")
    entry = import_module(module).run
    if not callable(entry):
        raise TypeError(f"Image generation entry is not callable: {module}.run")
    # Family application signatures are checked statically; importlib is the optional dependency boundary.
    generate = cast(ImageGeneration, entry)
    return generate(config, device=selected)
