"""Publish adapters through shared key/metadata assembly and model-owned layouts."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from unitunner.adapters.config import AdapterConfig
from unitunner.artifacts.formats.adapters import (
    AdapterTargetKeys,
    adapter_key_map,
    adapter_metadata,
    adapter_prefixes,
    convert_adapter_keys,
)
from unitunner.artifacts.tensors import save_tensors


def publish_adapter(
    path: Path,
    *,
    branches: nn.Module,
    target_names: Sequence[str],
    adapter: AdapterConfig,
    base_model: str,
    module_identifier: str,
    dtype: torch.dtype,
    namespace: Literal["transformer", "diffusion_model"] = "transformer",
    metadata: Mapping[str, str] | None = None,
    prefixes: Mapping[str, str] | None = None,
    published: Mapping[str, Tensor] | None = None,
) -> None:
    """Save ordinary mapped adapters or tensors from an explicit sliced-layout publisher."""
    if published is None:
        if prefixes is None:
            prefixes = adapter_prefixes(target_names, algorithm=adapter.algorithm, namespace=namespace)
        keys = adapter_key_map(
            [AdapterTargetKeys(str(index), prefixes[name]) for index, name in enumerate(target_names)],
            algorithm=adapter.algorithm,
        )
        published = convert_adapter_keys(branches.state_dict(), keys)
    info = adapter_metadata(
        algorithm=adapter.algorithm,
        base_model=base_model,
        rank=adapter.rank,
        alpha=adapter.alpha,
        lora_module_identifier=module_identifier,
    )
    info.update(metadata or {})
    save_tensors(published, path, metadata=info, dtype=dtype)
