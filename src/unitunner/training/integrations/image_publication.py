"""Publish image training results while preserving native adapter layouts and identities."""

from collections.abc import Mapping
from pathlib import Path

import torch

from unitunner.artifacts.formats.flux_adapters import flux_adapter_layout
from unitunner.artifacts.formats.krea2_adapters import krea2_adapter_prefixes
from unitunner.artifacts.formats.projection_slices import publish_sliced_adapter
from unitunner.config.training import TrainingConfig
from unitunner.models.flux.loading import save_flux
from unitunner.models.flux.transformer import FluxTransformer
from unitunner.models.flux2.loading import save_flux2
from unitunner.models.flux2.transformer import Flux2Transformer
from unitunner.models.flux3.loading import save_flux3
from unitunner.models.flux3.transformer import Flux3Transformer
from unitunner.models.image import ImageModel
from unitunner.models.krea2.loading import save_krea2
from unitunner.models.krea2.transformer import Krea2Transformer
from unitunner.models.moss_image.loading import save_moss_image
from unitunner.models.moss_image.transformer import MossImageTransformer
from unitunner.models.qwen_image.loading import save_qwen_image
from unitunner.models.qwen_image.transformer import QwenImageTransformer
from unitunner.models.qwen_image_2_1.loading import save_qwen_image_21
from unitunner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unitunner.models.z_image.loading import save_z_image
from unitunner.training.integrations.image_models import ImageBinding
from unitunner.training.parameters import ParameterInstallation
from unitunner.training.publishing import publish_adapter


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
    if adapter is None:
        if isinstance(model, FluxTransformer):
            save_flux(model, artifact, metadata=metadata, dtype=dtype)
        elif isinstance(model, Flux2Transformer):
            save_flux2(model, artifact, metadata=metadata, dtype=dtype)
        elif isinstance(model, Flux3Transformer):
            save_flux3(model, artifact, metadata=metadata, dtype=dtype)
        elif isinstance(model, Krea2Transformer):
            save_krea2(model, artifact, metadata=metadata, dtype=dtype)
        elif isinstance(model, MossImageTransformer):
            save_moss_image(model, artifact, metadata=metadata, dtype=dtype)
        elif isinstance(model, QwenImageTransformer):
            save_qwen_image(model, artifact, metadata=metadata, dtype=dtype)
        elif isinstance(model, QwenImage21Transformer):
            save_qwen_image_21(model, artifact, metadata=metadata, dtype=dtype)
        else:
            save_z_image(model, artifact, metadata=metadata, dtype=dtype)
    else:
        method_metadata = metadata
        metadata = {}
        if isinstance(model, Flux2Transformer):
            metadata["model_version"] = model.config.version
            metadata["model_variant"] = model.config.variant
        if isinstance(model, Flux3Transformer):
            metadata["flux3_model_version"] = model.config.variant
        if isinstance(model, QwenImageTransformer):
            metadata["qwen_image_model_version"] = model.config.variant
        if isinstance(model, QwenImage21Transformer):
            metadata["qwen_image_model_version"] = "2.1"
        metadata.update(method_metadata)
        if isinstance(model, FluxTransformer | Flux2Transformer):
            metadata["unitunner.adapter.layout"] = "flux-diffusers"
            targets = binding.adapter_targets()
            layout = flux_adapter_layout(
                {name: targets[name] for name in installation.plan.target_names},
                hidden_size=model.config.hidden_size,
                algorithm=adapter.algorithm,
            )
            published = publish_sliced_adapter(installation.branches.state_dict(), layout, algorithm=adapter.algorithm)
        else:
            published = None
        base_model, module_identifier = {
            "flux": ("flux", "networks.lora_flux"),
            "flux2": ("flux2", "networks.lora_flux2"),
            "flux3": ("flux3", "networks.lora_flux3"),
            "krea2": ("krea2", "networks.lora_krea2"),
            "moss-image": ("moss_image", "networks.lora_moss_image"),
            "qwen-image": ("qwen_image", "networks.lora_qwen_image"),
            "qwen-image-2-1": ("qwen_image_21", "networks.lora_qwen_image_21"),
            "z-image": ("z_image", "networks.lora_zimage"),
        }[config.model.family]
        if isinstance(model, FluxTransformer):
            base_model += f"_{model.config.variant}"
        elif isinstance(model, Flux2Transformer):
            base_model += f"_{model.config.version}"
        publish_adapter(
            artifact,
            branches=installation.branches,
            target_names=installation.plan.target_names,
            adapter=adapter,
            base_model=base_model,
            module_identifier=module_identifier,
            dtype=dtype,
            metadata=metadata,
            published=published,
            namespace="diffusion_model"
            if isinstance(model, Flux3Transformer | MossImageTransformer)
            else "transformer",
            prefixes=krea2_adapter_prefixes(installation.plan.target_names, algorithm=adapter.algorithm)
            if isinstance(model, Krea2Transformer)
            else None,
        )
