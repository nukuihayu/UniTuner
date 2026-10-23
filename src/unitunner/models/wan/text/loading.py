"""Strictly load and publish native Wan UMT5 encoder weights without tokenizer imports."""

from pathlib import Path
from typing import Literal

import torch

from unitunner.artifacts.tensors import TensorReader, iter_tensors, save_tensors
from unitunner.config.text import DEFAULT_TEXT_CONFIG, TextEncodingConfig
from unitunner.models.components.text_loading import load_text_state, prepare_text_model, text_dtype
from unitunner.models.wan.config.t5 import WanT5Config
from unitunner.models.wan.text.model import WanT5Encoder

_CONFIG_KEY = "unitunner.wan.t5_config"
_CPU = torch.device("cpu")


def save_wan_t5(model: WanT5Encoder, path: Path) -> None:
    """Atomically publish exact native keys and authoritative immutable encoder topology."""
    save_tensors(model.state_dict(), path, metadata={_CONFIG_KEY: model.config.model_dump_json()})


def load_wan_t5(
    path: Path,
    *,
    config: WanT5Config | None = None,
    prefix: Literal["", "encoder."] = "",
    device: torch.device = _CPU,
    text_config: TextEncodingConfig = DEFAULT_TEXT_CONFIG,
) -> WanT5Encoder:
    """Load owned frozen/eval FP32 encoder tensors from native single/sharded safetensors.

    External tensors require explicit topology. Remove only an explicitly
    selected uniform anchored prefix; reject mixed keys and substring aliases.
    Construction on meta preserves RNG. Validate all names/shapes/finite values
    before assigning parameters; no source file is modified or imported.
    """
    if prefix not in ("", "encoder."):
        raise ValueError("Wan T5 prefix must be empty or encoder.")
    with TensorReader(path) as reader:
        serialized = reader.metadata().get(_CONFIG_KEY)
    embedded = None if serialized is None else WanT5Config.model_validate_json(serialized)
    if config is None:
        if embedded is None:
            raise ValueError("Wan T5 external weights require explicit topology")
        config = embedded
    elif embedded is not None and config != embedded:
        raise ValueError("Wan T5 explicit topology differs from artifact metadata")
    with torch.device("meta"):
        model = WanT5Encoder(config)
    tensors = dict(iter_tensors(path))
    if prefix:
        if any(not name.startswith(prefix) for name in tensors):
            raise ValueError("Wan T5 tensors must all use the selected anchored prefix")
        tensors = {name.removeprefix(prefix): value for name, value in tensors.items()}
    dtype = text_dtype(text_config, torch.float32)
    load_text_state(model, tensors, dtype=dtype, label="Wan T5")
    prepare_text_model(model, text_config, family="umt5", device=device)
    return model.eval().requires_grad_(False)
