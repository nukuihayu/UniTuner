"""Publish LoRA in official Diffusers/PEFT names with exact effective scaling."""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from unitunner.artifacts.formats.transformer import official_projection_sources
from unitunner.models.catalog import ModelFamily


def diffusers_lora_state(state: Mapping[str, Tensor], target_names: Sequence[str], *, family: str) -> dict[str, Tensor]:
    """Split fused native output factors and bake alpha/r into PEFT's B factor.

    The portable Diffusers loader initializes vanilla LoRA with alpha=rank.
    Baking the original scaling preserves delta without depending on consumer
    interpretation of Kohya alpha tensors or descriptive safetensors metadata.
    No dense update, model copy or SVD is needed; source factors stay untouched.
    """
    expected = {f"{i}.{part}" for i in range(len(target_names)) for part in ("down", "up", "alpha")}
    ModelFamily(family)
    if state.keys() != expected:
        raise ValueError("Diffusers export requires complete vanilla LoRA factors")
    result = {}
    for index, name in enumerate(target_names):
        down, up, alpha = (state[f"{index}.{part}"] for part in ("down", "up", "alpha"))
        paths = official_projection_sources(family, name + ".weight")
        rows = up.shape[0]
        if len(paths) == 1:
            boundaries = [(0, rows)]
        elif len(paths) == 3 and rows % 3 == 0:
            third = rows // 3
            boundaries = [(i * third, (i + 1) * third) for i in range(3)]
        elif len(paths) == 4 and rows > 3 * down.shape[1]:
            width = down.shape[1]
            boundaries = [(i * width, (i + 1) * width) for i in range(3)] + [(3 * width, rows)]
        else:
            raise ValueError(f"Unsupported official LoRA projection boundaries: {name}")
        compute_dtype = torch.float64 if up.dtype == torch.float64 else torch.float32
        scaled_up = up.to(compute_dtype) * (alpha.to(compute_dtype) / down.shape[0])
        if family == "minimax-h3" and name.endswith(".mlp.fc1"):
            if rows % 2:
                raise ValueError("MiniMaxH3 SwiGLU output width must be even")
            gate, value = scaled_up.chunk(2, dim=0)
            scaled_up = torch.cat((value, gate), dim=0)
        for path, (start, stop) in zip(paths, boundaries, strict=True):
            prefix = "transformer." + path.removesuffix(".weight")
            a_key, b_key = prefix + ".lora_A.weight", prefix + ".lora_B.weight"
            if a_key in result or b_key in result:
                raise ValueError(f"Duplicate official LoRA projection: {path}")
            result[a_key], result[b_key] = down, scaled_up[start:stop]
    return result
