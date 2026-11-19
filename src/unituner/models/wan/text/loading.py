"""Strictly load and publish native Wan UMT5 encoder weights without tokenizer imports."""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from unituner.artifacts.tensors import TensorReader, iter_tensors, save_tensors
from unituner.config.text import DEFAULT_TEXT_CONFIG, TextEncodingConfig
from unituner.models.components.text_loading import load_text_state, prepare_text_model, text_dtype
from unituner.models.wan.config.t5 import WanT5Config
from unituner.models.wan.text.model import WanT5Encoder

_CONFIG_KEY = "unituner.wan.t5_config"
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

    External native tensors infer topology from keys/shapes when metadata is absent.
    Automatically recognize a uniform encoder. prefix, or remove the explicitly
    selected anchored prefix; reject mixed keys and substring aliases.
    Construction on meta preserves RNG. Validate all names/shapes/finite values
    before assigning parameters; no source file is modified or imported.
    """
    if prefix not in ("", "encoder."):
        raise ValueError("Wan T5 prefix must be empty or encoder.")
    with TensorReader(path) as reader:
        serialized = reader.metadata().get(_CONFIG_KEY)
    embedded = None if serialized is None else WanT5Config.model_validate_json(serialized)
    if config is not None and embedded is not None and config != embedded:
        raise ValueError("Wan T5 explicit topology differs from artifact metadata")
    tensors = dict(iter_tensors(path))
    if not prefix and tensors and all(name.startswith("encoder.") for name in tensors):
        prefix = "encoder."
    if prefix:
        if any(not name.startswith(prefix) for name in tensors):
            raise ValueError("Wan T5 tensors must all use the selected anchored prefix")
        tensors = {name.removeprefix(prefix): value for name, value in tensors.items()}
    if config is None:
        config = embedded if embedded is not None else _infer_t5_config(tensors)
    with torch.device("meta"):
        model = WanT5Encoder(config)
    dtype = text_dtype(text_config, torch.float32)
    load_text_state(model, tensors, dtype=dtype, label="Wan T5")
    prepare_text_model(model, text_config, family="umt5", device=device)
    return model.eval().requires_grad_(False)


def _infer_t5_config(tensors: Mapping[str, torch.Tensor]) -> WanT5Config:
    """Read native geometry; full loader validation still checks every key and shape."""
    layers = {int(match[1]) for key in tensors if (match := re.match(r"blocks\.(\d+)\.", key))}
    if not layers or layers != set(range(max(layers) + 1)):
        raise ValueError("Wan T5 external topology requires contiguous native blocks")
    shared = "pos_embedding.embedding.weight" in tensors
    bias_key = "pos_embedding.embedding.weight" if shared else "blocks.0.pos_embedding.embedding.weight"
    try:
        vocab, hidden = tensors["token_embedding.weight"].shape
        attention, _ = tensors["blocks.0.attn.q.weight"].shape
        ffn, _ = tensors["blocks.0.ffn.fc1.weight"].shape
        buckets, heads = tensors[bias_key].shape
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Cannot infer Wan T5 external topology; supply explicit config for non-native weights"
        ) from error
    return WanT5Config(
        vocab_size=vocab,
        hidden_size=hidden,
        attention_dim=attention,
        ffn_dim=ffn,
        num_heads=heads,
        num_layers=len(layers),
        num_buckets=buckets,
        shared_position_bias=shared,
    )
