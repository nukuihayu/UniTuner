"""Publish image training results while preserving native adapter layouts and identities."""

from collections.abc import Mapping
from pathlib import Path

import torch

from unituner.artifacts.formats.flux_adapters import flux_adapter_layout
from unituner.artifacts.formats.krea2_adapters import krea2_adapter_prefixes
from unituner.artifacts.formats.projection_slices import publish_sliced_adapter
from unituner.config.training import TrainingConfig
from unituner.models.flux.transformer import FluxTransformer
from unituner.models.flux2.transformer import Flux2Transformer
from unituner.models.flux3.transformer import Flux3Transformer
from unituner.models.image import ImageModel, model_spec, save_image_model
from unituner.models.krea2.transformer import Krea2Transformer
from unituner.models.qwen_image.transformer import QwenImageTransformer
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unituner.training.integrations.image_models import ImageBinding
from unituner.training.parameters import ParameterInstallation
from unituner.training.publishing import publish_adapter


def publish_image_model(
    config: TrainingConfig,
    model: ImageModel,
    binding: ImageBinding,
    installation: ParameterInstallation,
    artifact: Path,
    *,
    dtype: torch.dtype,
    metadata: Mapping[str, str],
) -> None:
    """Save a native model or a consumer-compatible adapter without changing live weights."""
    adapter = config.train.adapter
    spec = model_spec(model)
    if adapter is None:
        save_image_model(model, artifact, metadata=metadata, dtype=dtype)
    else:
        method_metadata = metadata
        metadata = {}
        if isinstance(model, Flux2Transformer):
            metadata["model_version"] = model.config.version
            metadata["model_variant"] = config.model.variant if spec.native_variant is not None else model.config.variant
        if isinstance(model, Flux3Transformer):
            metadata["flux3_model_version"] = model.config.variant
        if isinstance(model, QwenImageTransformer):
            metadata["qwen_image_model_version"] = model.config.variant
        if isinstance(model, QwenImage21Transformer):
            metadata["qwen_image_model_version"] = "2.1"
        metadata.update(method_metadata)
        if isinstance(model, FluxTransformer | Flux2Transformer) and config.output.adapter_format == "comfyui":
            metadata["unituner.adapter.layout"] = "flux-diffusers"
            targets = binding.adapter_targets()
            layout = flux_adapter_layout(
                {name: targets[name] for name in installation.plan.target_names},
                hidden_size=model.config.hidden_size,
                algorithm=adapter.algorithm,
            )
            published = publish_sliced_adapter(installation.branches.state_dict(), layout, algorithm=adapter.algorithm)
        else:
            published = None
        base_model = spec.adapter_base
        if isinstance(model, FluxTransformer):
            base_model += f"_{model.config.variant}"
        elif isinstance(model, Flux2Transformer) and spec.native_version is None:
            base_model += f"_{model.config.version}"
        publish_adapter(
            artifact,
            branches=installation.branches,
            target_names=installation.plan.target_names,
            adapter=adapter,
            base_model=base_model,
            module_identifier=spec.adapter_module,
            dtype=dtype,
            adapter_format=config.output.adapter_format,
            family=spec.backbone,
            metadata=metadata,
            published=published,
            namespace=spec.adapter_namespace,
            prefixes=krea2_adapter_prefixes(installation.plan.target_names, algorithm=adapter.algorithm)
            if isinstance(model, Krea2Transformer)
            else None,
        )
