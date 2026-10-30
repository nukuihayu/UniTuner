"""Map native fused FLUX.1/FLUX.2 adapter rows to ComfyUI Diffusers projections."""

import re
from collections.abc import Mapping
from typing import Literal

from torch import nn

from unituner.artifacts.formats.adapters import adapter_prefixes
from unituner.artifacts.formats.projection_slices import AdapterProjectionSlice, validate_projection_slices


def flux_adapter_layout(
    targets: Mapping[str, nn.Linear], *, hidden_size: int, algorithm: Literal["lora", "loha", "lokr"]
) -> dict[str, tuple[AdapterProjectionSlice, ...]]:
    """Map every selected native projection, preserving QKV order and fused single blocks.

    ComfyUI flux_to_diffusers supplies Q/K/V output-row offsets and the whole
    single attn.to_qkv_mlp_proj alias for both Flux and Flux2. No model weights,
    training factors or optimizer references are changed. Unsupported native
    projection paths fail instead of publishing an unrecognized prefix.
    """
    result: dict[str, tuple[AdapterProjectionSlice, ...]] = {}
    direct = {
        "img_attn.proj": "attn.to_out.0",
        "txt_attn.proj": "attn.to_add_out",
        "img_mlp.0": "ff.net.0.proj",
        "img_mlp.2": "ff.net.2",
        "txt_mlp.0": "ff_context.net.0.proj",
        "txt_mlp.2": "ff_context.net.2",
    }
    for name, projection in targets.items():
        paths: list[tuple[str, int, int]] = []
        if match := re.fullmatch(r"double_blocks\.(\d+)\.(.+)", name):
            index, tail = match.groups()
            prefix = f"transformer_blocks.{index}."
            if tail in ("img_attn.qkv", "txt_attn.qkv"):
                if projection.out_features != 3 * hidden_size:
                    raise ValueError("FLUX QKV output must equal three hidden widths")
                fields = (
                    ("to_q", "to_k", "to_v") if tail == "img_attn.qkv" else ("add_q_proj", "add_k_proj", "add_v_proj")
                )
                paths = [
                    (prefix + "attn." + field, i * hidden_size, (i + 1) * hidden_size) for i, field in enumerate(fields)
                ]
            elif tail in direct:
                paths = [(prefix + direct[tail], 0, projection.out_features)]
        elif match := re.fullmatch(r"single_blocks\.(\d+)\.(linear1|linear2)", name):
            index, tail = match.groups()
            field = "attn.to_qkv_mlp_proj" if tail == "linear1" else "attn.to_out"
            paths = [(f"single_transformer_blocks.{index}.{field}", 0, projection.out_features)]
        if not paths:
            raise ValueError(f"Unsupported FLUX adapter projection: {name}")
        prefixes = adapter_prefixes((path for path, _, _ in paths), algorithm=algorithm)
        parts = tuple(AdapterProjectionSlice(prefixes[path], start, stop) for path, start, stop in paths)
        validate_projection_slices(parts, projection.out_features)
        result[name] = parts
    if not result or len({part.external for parts in result.values() for part in parts}) != sum(
        map(len, result.values())
    ):
        raise ValueError("FLUX adapter layout requires unique nonempty consumer projections")
    return result
