"""Convert explicit floating MOSS artifact layouts without runtime format guessing."""

from collections.abc import Mapping
from typing import Literal

import torch
from torch import Tensor

from unituner.models.moss_image.config.model import MossImageConfig
from unituner.models.moss_image.layers import RotaryEmbedding


def convert_transformer_tensors(
    tensors: Mapping[str, Tensor],
    config: MossImageConfig,
    *,
    prefix: Literal["", "model.diffusion_model.", "diffusion_model."] = "",
    fused_qkv: bool = False,
) -> dict[str, Tensor]:
    """Return native keys, splitting main QKV rows only when explicitly requested.

    Every input key must use the selected prefix. Fused main weights have shape
    [3*dim,dim], ordered Q then K then V; caption QKV stays fused. No quantized
    scales/metadata or mixed main fused/split layouts are accepted. Other tensors
    are borrowed unchanged. The loader subsequently validates complete keys,
    shapes and finite values before assignment. Rebuild the derived FP32 RoPE
    bands from topology, accepting absent or finite rounded source bands only.
    """
    if prefix not in ("", "model.diffusion_model.", "diffusion_model."):
        raise ValueError("Unsupported MOSS artifact prefix")
    converted: dict[str, Tensor] = {}
    for original, value in tensors.items():
        if not original.startswith(prefix):
            raise ValueError(f"MOSS artifact key lacks selected prefix: {original}")
        key = original[len(prefix) :]
        if key == "rope.bands":
            if (
                value.shape != (config.dim // config.num_heads // 8,)
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Invalid derived MOSS RoPE bands")
            continue
        if key.startswith("layers.") and ".proj.qkv." in key:
            if not fused_qkv or not key.endswith(".proj.qkv.weight") or value.shape != (3 * config.dim, config.dim):
                raise ValueError(f"Invalid or unselected MOSS fused QKV: {original}")
            parts = [
                (key.replace(".proj.qkv.", f".proj.{part}."), chunk.contiguous())
                for part, chunk in zip(("q", "k", "v"), value.chunk(3, dim=0), strict=True)
            ]
        else:
            if fused_qkv and key.startswith("layers.") and any(f".proj.{part}." in key for part in ("q", "k", "v")):
                raise ValueError(f"MOSS fused QKV layout contains split projection: {original}")
            parts = [(key, value)]
        for native, tensor in parts:
            if native in converted:
                raise ValueError(f"Duplicate converted MOSS tensor: {native}")
            converted[native] = tensor
    converted["rope.bands"] = RotaryEmbedding(config.dim // config.num_heads, config.rope_base).default_bands(
        torch.device("cpu")
    )
    return converted
