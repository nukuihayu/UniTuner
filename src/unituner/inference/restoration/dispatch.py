"""Select native restoration by declared model family at the CLI boundary."""

from unituner.config.restoration_inference import RestorationInferenceConfig
from unituner.models.catalog import image_restoration_provider
from unituner.models.providers import load_provider


def run(config: RestorationInferenceConfig, *, device: str) -> int:
    return load_provider(image_restoration_provider(config.model.family))(config, device=device)
