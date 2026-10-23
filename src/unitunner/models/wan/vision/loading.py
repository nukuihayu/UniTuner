"""Strictly validate complete Wan CLIP artifacts while materializing only visual weights."""

from pathlib import Path
from typing import Literal

import torch

from unitunner.artifacts.tensors import TensorReader, iter_tensors, save_tensors
from unitunner.models.wan.config.clip import WanClipConfig, WanClipTextArtifactConfig
from unitunner.models.wan.vision.model import WanClipVision

_CONFIG_KEY = "unitunner.wan.clip_config"
_CPU = torch.device("cpu")


def save_wan_clip(model: WanClipVision, path: Path) -> None:
    """Atomically save all visual tensors and authoritative immutable native topology."""
    save_tensors(model.state_dict(), path, metadata={_CONFIG_KEY: model.config.model_dump_json()})


def _text_shapes(config: WanClipTextArtifactConfig, output_dim: int) -> dict[str, tuple[int, ...]]:
    """Inventory the native unused XLM-Roberta branch, including its two-layer head."""
    width = config.hidden_size
    middle = (width + output_dim) // 2
    result = {
        "textual.token_embedding.weight": (config.vocab_size, width),
        "textual.type_embedding.weight": (config.type_size, width),
        "textual.pos_embedding.weight": (config.max_length, width),
        "textual.norm.weight": (width,),
        "textual.norm.bias": (width,),
        "textual.head.0.weight": (middle, width),
        "textual.head.2.weight": (output_dim, middle),
        "log_scale": (),
    }
    for index in range(config.num_layers):
        prefix = f"textual.blocks.{index}."
        for name in ("q", "k", "v", "o"):
            result[prefix + f"attn.{name}.weight"] = (width, width)
            result[prefix + f"attn.{name}.bias"] = (width,)
        for name in ("norm1", "norm2"):
            result[prefix + name + ".weight"] = (width,)
            result[prefix + name + ".bias"] = (width,)
        result[prefix + "ffn.0.weight"] = (4 * width, width)
        result[prefix + "ffn.0.bias"] = (4 * width,)
        result[prefix + "ffn.2.weight"] = (width, 4 * width)
        result[prefix + "ffn.2.bias"] = (width,)
    return result


def load_wan_clip(
    path: Path,
    *,
    config: WanClipConfig | None = None,
    source_format: Literal["native", "visual", "full"] = "native",
    text_artifact_config: WanClipTextArtifactConfig | None = None,
    device: torch.device = _CPU,
) -> WanClipVision:
    """Load frozen/eval owned FP32 visual parameters from single/sharded safetensors.

    Native and visual formats use unprefixed visual keys. Full format requires
    exactly visual.* plus the complete native XLM-Roberta/log_scale inventory;
    validate every unused tensor before discarding it. No unused text module or
    tokenizer is constructed. Meta construction preserves RNG. External visual
    topology must be explicit; full text topology defaults to published Wan.
    """
    if source_format not in ("native", "visual", "full"):
        raise ValueError("Wan CLIP source_format must be native, visual or full")
    if text_artifact_config is not None and source_format != "full":
        raise ValueError("Wan CLIP text_artifact_config requires full source_format")
    with TensorReader(path) as reader:
        serialized = reader.metadata().get(_CONFIG_KEY)
    embedded = None if serialized is None else WanClipConfig.model_validate_json(serialized)
    if config is None:
        if embedded is None:
            raise ValueError("Wan CLIP external weights require explicit visual topology")
        config = embedded
    elif embedded is not None and config != embedded:
        raise ValueError("Wan CLIP explicit topology differs from artifact metadata")
    with torch.device("meta"):
        model = WanClipVision(config)
    prefix = "visual." if source_format == "full" else ""
    expected = {prefix + name: tuple(value.shape) for name, value in model.state_dict().items()}
    if source_format == "full":
        expected.update(_text_shapes(text_artifact_config or WanClipTextArtifactConfig(), config.output_dim))
    visual = {}
    seen: set[str] = set()
    for name, value in iter_tensors(path):
        if name not in expected:
            raise ValueError(f"Wan CLIP unexpected tensor: {name}")
        if (
            tuple(value.shape) != expected[name]
            or value.dtype not in (torch.float64, torch.float32, torch.float16, torch.bfloat16)
            or not torch.isfinite(value).all()
            or not torch.isfinite(value.float()).all()
        ):
            raise ValueError(f"Wan CLIP invalid tensor shape/type/values: {name}")
        seen.add(name)
        if not prefix or name.startswith(prefix):
            visual[name.removeprefix(prefix)] = value.to(device=device, dtype=torch.float32, copy=True)
    if seen != expected.keys():
        raise ValueError(f"Wan CLIP missing tensors: {sorted(expected.keys() - seen)}")
    model.load_state_dict(visual, strict=True, assign=True)
    return model.eval().requires_grad_(False)
