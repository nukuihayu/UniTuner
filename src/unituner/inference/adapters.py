"""Scope native image adapter loading and release all borrowed projection references."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from unituner.adapters.loading import LoadedAdapter
from unituner.artifacts.formats.adapters import adapter_prefixes
from unituner.artifacts.formats.flux_adapters import flux_adapter_layout
from unituner.artifacts.formats.krea2_adapters import krea2_adapter_prefixes
from unituner.config.generation import GenerationAdapterConfig
from unituner.models.catalog import ModelFamily
from unituner.models.image import ImageModel, image_adapter_targets, model_spec


@contextmanager
def image_adapter(model: ImageModel, config: GenerationAdapterConfig | None) -> Iterator[None]:
    """Attach selected published adapters only for this scope, including on failure."""
    if config is None:
        yield
        return
    targets = image_adapter_targets(model)
    spec = model_spec(model)
    layout = None
    prefixes = None
    if spec.backbone in (ModelFamily.FLUX, ModelFamily.FLUX2):
        layout = flux_adapter_layout(targets, hidden_size=model.config.hidden_size, algorithm=config.config.algorithm)
    elif spec.backbone == ModelFamily.KREA2:
        prefixes = krea2_adapter_prefixes(targets, algorithm=config.config.algorithm)
    else:
        prefixes = adapter_prefixes(
            targets,
            algorithm=config.config.algorithm,
            namespace=spec.adapter_namespace,
        )
    owner = LoadedAdapter(
        Path(config.weights), config.config, targets, projection_layout=layout, external_prefixes=prefixes
    )
    try:
        yield
    finally:
        owner.close()
