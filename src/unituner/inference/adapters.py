"""Scope native image adapter loading and release all borrowed projection references."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from torch import nn

from unituner.adapters.loading import LoadedAdapter
from unituner.artifacts.formats.adapters import adapter_prefixes
from unituner.artifacts.formats.flux_adapters import flux_adapter_layout
from unituner.artifacts.formats.krea2_adapters import krea2_adapter_prefixes
from unituner.config.generation import GenerationAdapterConfig
from unituner.models.flux.projections import adapter_projections as flux_targets
from unituner.models.flux.transformer import FluxTransformer
from unituner.models.flux2.projections import adapter_projections as flux2_targets
from unituner.models.flux2.transformer import Flux2Transformer
from unituner.models.flux3.projections import adapter_projections as flux3_targets
from unituner.models.flux3.transformer import Flux3Transformer
from unituner.models.image import ImageModel
from unituner.models.krea2.projections import adapter_projections as krea2_targets
from unituner.models.krea2.transformer import Krea2Transformer
from unituner.models.moss_image.projections import adapter_projections as moss_targets
from unituner.models.moss_image.transformer import MossImageTransformer
from unituner.models.qwen_image.projections import adapter_projections as qwen_targets
from unituner.models.qwen_image.transformer import QwenImageTransformer
from unituner.models.qwen_image_2_1.projections import adapter_projections as qwen21_targets
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unituner.models.z_image.projections import adapter_projections as z_image_targets
from unituner.models.z_image.transformer import ZImageTransformer


def _adapter_targets(model: ImageModel) -> dict[str, nn.Linear]:
    if isinstance(model, FluxTransformer):
        return flux_targets(model)
    if isinstance(model, Flux2Transformer):
        return flux2_targets(model)
    if isinstance(model, Flux3Transformer):
        return flux3_targets(model)
    if isinstance(model, Krea2Transformer):
        return krea2_targets(model)
    if isinstance(model, MossImageTransformer):
        return moss_targets(model)
    if isinstance(model, QwenImageTransformer):
        return qwen_targets(model)
    if isinstance(model, QwenImage21Transformer):
        return qwen21_targets(model)
    if isinstance(model, ZImageTransformer):
        return z_image_targets(model)
    raise TypeError(f"No image adapter targets for {type(model).__name__}")


@contextmanager
def image_adapter(model: ImageModel, config: GenerationAdapterConfig | None) -> Iterator[None]:
    """Attach selected published adapters only for this scope, including on failure."""
    if config is None:
        yield
        return
    targets = _adapter_targets(model)
    layout = None
    prefixes = None
    if isinstance(model, FluxTransformer | Flux2Transformer):
        layout = flux_adapter_layout(targets, hidden_size=model.config.hidden_size, algorithm=config.config.algorithm)
    elif isinstance(model, Krea2Transformer):
        prefixes = krea2_adapter_prefixes(targets, algorithm=config.config.algorithm)
    else:
        prefixes = adapter_prefixes(
            targets,
            algorithm=config.config.algorithm,
            namespace="diffusion_model"
            if isinstance(model, Flux3Transformer | MossImageTransformer)
            else "transformer",
        )
    owner = LoadedAdapter(
        Path(config.weights), config.config, targets, projection_layout=layout, external_prefixes=prefixes
    )
    try:
        yield
    finally:
        owner.close()
