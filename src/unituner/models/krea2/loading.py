"""Strictly load and atomically publish native Krea2 model artifacts."""

from collections.abc import Mapping
from pathlib import Path

import torch

from unituner.artifacts.formats.transformer import convert_transformer
from unituner.artifacts.tensors import save_tensors
from unituner.models.components.state_loading import load_model_state
from unituner.models.components.transformer_artifacts import (
    resolve_transformer_config,
    transformer_tensors,
)
from unituner.models.krea2.config.model import Krea2Config
from unituner.models.krea2.transformer import Krea2Transformer

_CONFIG_KEY = "unituner.krea2.config"
_CPU = torch.device("cpu")


def save_krea2(
    model: Krea2Transformer, path: Path, *, metadata: Mapping[str, str] | None = None, dtype: torch.dtype | None = None
) -> None:
    """Atomically export native weights and topology, casting only published floating tensors."""
    save_tensors(
        model.state_dict(),
        path,
        dtype=dtype,
        metadata={**({} if metadata is None else metadata), _CONFIG_KEY: model.config.model_dump_json()},
    )


def load_krea2(
    path: Path,
    *,
    config: Krea2Config | None = None,
    variant: str | None = None,
    version: str | None = None,
    device: torch.device = _CPU,
    dtype: torch.dtype = torch.float32,
) -> Krea2Transformer:
    """Load native, Diffusers or ComfyUI weights with validated topology, keys and shapes."""
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Krea2 weights require FP32/FP16/BF16 loading")
    config = resolve_transformer_config(
        path,
        Krea2Config,
        metadata_key=_CONFIG_KEY,
        config=config,
        variant=variant,
        version=version,
        mismatch_message="Krea2 explicit topology differs from artifact metadata",
    )
    with torch.device("meta"):
        model = Krea2Transformer(config)
    expected, tensors = model.state_dict(), transformer_tensors(path)
    tensors = convert_transformer("krea2", tensors, expected)
    load_model_state(model, tensors, device=device, dtype=dtype, label="Krea2")
    return model
