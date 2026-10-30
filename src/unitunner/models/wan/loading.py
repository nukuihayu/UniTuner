"""Strictly load and atomically publish native Wan model artifacts."""

from collections.abc import Mapping
from pathlib import Path

import torch

from unitunner.artifacts.formats.transformer import convert_transformer
from unitunner.artifacts.tensors import save_tensors
from unitunner.models.components.state_loading import load_model_state
from unitunner.models.components.transformer_artifacts import (
    resolve_transformer_config,
    transformer_tensors,
)
from unitunner.models.wan.config.model import WanConfig
from unitunner.models.wan.transformer import WanTransformer

_CONFIG_KEY = "unitunner.wan.config"
_CPU = torch.device("cpu")


def save_wan(
    model: WanTransformer, path: Path, *, metadata: Mapping[str, str] | None = None, dtype: torch.dtype | None = None
) -> None:
    """Atomically export native weights and topology, casting only published floating tensors."""
    save_tensors(
        model.state_dict(),
        path,
        dtype=dtype,
        metadata={**({} if metadata is None else metadata), _CONFIG_KEY: model.config.model_dump_json()},
    )


def load_wan(
    path: Path,
    *,
    config: WanConfig | None = None,
    variant: str | None = None,
    version: str | None = None,
    device: torch.device = _CPU,
    dtype: torch.dtype = torch.float32,
) -> WanTransformer:
    """Load native, Diffusers or ComfyUI weights with validated topology, keys and shapes."""
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Wan transformer requires FP32/FP16/BF16 loading")
    config = resolve_transformer_config(
        path,
        WanConfig,
        metadata_key=_CONFIG_KEY,
        config=config,
        variant=variant,
        version=version,
        mismatch_message="Wan explicit topology differs from artifact metadata",
    )
    with torch.device("meta"):
        model = WanTransformer(config)
    expected, tensors = model.state_dict(), transformer_tensors(path)
    tensors = convert_transformer("wan", tensors, expected)

    def storage_dtype(name: str) -> torch.dtype:
        protected = name.startswith(("time_embedding.", "time_projection.")) or name.endswith(".modulation")
        return torch.float32 if protected else dtype

    load_model_state(
        model,
        tensors,
        device=device,
        dtype=dtype,
        label="Wan",
        floating_dtypes={name: storage_dtype(name) for name in expected},
    )
    return model
