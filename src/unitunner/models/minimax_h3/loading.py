"""Strictly load and atomically publish native MiniMaxH3 AV denoiser artifacts."""

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
from unitunner.models.minimax_h3.config.model import MiniMaxH3Config
from unitunner.models.minimax_h3.transformer import MiniMaxH3Transformer

_CONFIG_KEY = "unitunner.minimax_h3.config"
_CPU = torch.device("cpu")


def save_minimax_h3(
    model: MiniMaxH3Transformer,
    path: Path,
    *,
    metadata: Mapping[str, str] | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    """Atomically export native weights and topology, casting only published floating tensors."""
    save_tensors(
        model.state_dict(),
        path,
        dtype=dtype,
        metadata={**({} if metadata is None else metadata), _CONFIG_KEY: model.config.model_dump_json()},
    )


def load_minimax_h3(
    path: Path,
    *,
    config: MiniMaxH3Config | None = None,
    variant: str | None = None,
    version: str | None = None,
    device: torch.device = _CPU,
    dtype: torch.dtype = torch.float32,
) -> MiniMaxH3Transformer:
    """Load native, Diffusers or ComfyUI weights with validated topology, keys and shapes."""
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("MiniMaxH3 transformer requires FP32/FP16/BF16 loading")
    config = resolve_transformer_config(
        path,
        MiniMaxH3Config,
        metadata_key=_CONFIG_KEY,
        config=config,
        variant=variant,
        version=version,
        mismatch_message="MiniMaxH3 explicit topology differs from artifact metadata",
    )
    with torch.device("meta"):
        model = MiniMaxH3Transformer(config, dtype=dtype)
    expected, tensors = model.state_dict(), transformer_tensors(path)
    tensors = convert_transformer("minimax_h3", tensors, expected)
    load_model_state(
        model,
        tensors,
        device=device,
        dtype=dtype,
        label="MiniMaxH3",
        floating_dtypes={name: value.dtype for name, value in expected.items()},
    )
    return model
