"""Scope native image adapter loading and release all borrowed projection references."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from torch import nn

from unituner.adapters.loading import LoadedAdapter
from unituner.artifacts.formats.adapters import adapter_prefixes
from unituner.artifacts.formats.flux_adapters import flux_adapter_layout
from unituner.artifacts.formats.krea2_adapters import krea2_adapter_prefixes
from unituner.config.generation import GenerationAdapterConfig
from unituner.models.catalog import ModelFamily
from unituner.models.image import ImageModel, image_adapter_targets, model_spec


@contextmanager
def image_adapter(
    model: ImageModel, config: GenerationAdapterConfig | None, *,
    targets: Mapping[str, nn.Linear] | None = None, namespace: str = "diffusion_model",
) -> Iterator[None]:
    """Attach selected published adapters only for this scope, including on failure."""
    if config is None:
        yield
        return
    custom = targets is not None
    if targets is None:
        targets = image_adapter_targets(model)
    spec = None if custom else model_spec(model)
    layout = None
    prefixes = None
    if spec is not None and spec.backbone in (ModelFamily.FLUX, ModelFamily.FLUX2):
        layout = flux_adapter_layout(targets, hidden_size=model.config.hidden_size, algorithm=config.config.algorithm)
    elif spec is not None and spec.backbone == ModelFamily.KREA2:
        prefixes = krea2_adapter_prefixes(targets, algorithm=config.config.algorithm)
    else:
        prefixes = adapter_prefixes(
            targets,
            algorithm=config.config.algorithm,
            namespace=namespace if spec is None else spec.adapter_namespace,
        )
    owner = LoadedAdapter(
        Path(config.weights), config.config, targets, projection_layout=layout, external_prefixes=prefixes
    )
    try:
        yield
    finally:
        owner.close()
