"""Map Z-Image VAE artifact keys explicitly without importing model implementations."""

import re
from collections.abc import Mapping, Sequence

from torch import Tensor


def vae_key_map(native_keys: Sequence[str], *, num_blocks: int) -> dict[str, str]:
    """Map native/Diffusers VAE names bijectively to ComfyUI VAE names.

    Decoder block order is reversed using the supplied actual topology depth,
    rather than assuming four blocks. Numerics and live parameters are untouched.
    """
    if num_blocks < 1:
        raise ValueError("VAE block count must be positive")
    mapping: dict[str, str] = {}
    for key in native_keys:
        external = key.replace(".conv_norm_out.", ".norm_out.")
        external = re.sub(r"encoder\.down_blocks\.(\d+)\.resnets\.", r"encoder.down.\1.block.", external)
        external = re.sub(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0\.", r"encoder.down.\1.downsample.", external)
        match = re.match(r"decoder\.up_blocks\.(\d+)\.(.*)", external)
        if match is not None:
            index, suffix = int(match[1]), match[2]
            if not 0 <= index < num_blocks:
                raise ValueError("VAE decoder block index exceeds topology")
            external = f"decoder.up.{num_blocks - 1 - index}." + suffix.replace("resnets.", "block.").replace(
                "upsamplers.0.", "upsample."
            )
        external = external.replace(".mid_block.resnets.0.", ".mid.block_1.").replace(
            ".mid_block.resnets.1.", ".mid.block_2."
        )
        if ".mid_block.attentions.0." in external:
            external = external.replace(".mid_block.attentions.0.", ".mid.attn_1.")
            for native, published in (
                ("group_norm", "norm"),
                ("to_q", "q"),
                ("to_k", "k"),
                ("to_v", "v"),
                ("to_out.0", "proj_out"),
            ):
                external = external.replace(f".attn_1.{native}.", f".attn_1.{published}.")
        external = external.replace(".conv_shortcut.", ".nin_shortcut.")
        if key in mapping or external in mapping.values():
            raise ValueError("VAE key mapping must be bijective")
        mapping[key] = external
    return mapping


def convert_vae_tensors(
    tensors: Mapping[str, Tensor], keys: Mapping[str, str], *, to_native: bool
) -> dict[str, Tensor]:
    """Convert all VAE keys and the reversible 1x1-convolution/linear attention layout.

    External attention projection weights may use [O,I,1,1] or [O,I]. Native
    output always uses [O,I]; export uses [O,I,1,1]. Values share input storage.
    Missing, extra or colliding keys fail before any destination is published.
    """
    if len(keys) != len(set(keys.values())):
        raise ValueError("VAE key mapping must be bijective")
    expected = set(keys.values()) if to_native else set(keys)
    if set(tensors) != expected:
        raise ValueError(
            f"VAE keys differ: missing={sorted(expected - tensors.keys())}, extra={sorted(tensors.keys() - expected)}"
        )
    result: dict[str, Tensor] = {}
    for native, external in keys.items():
        value = tensors[external if to_native else native]
        projection = ".attentions.0.to_" in native and native.endswith(".weight")
        if projection:
            if to_native:
                if value.ndim == 4 and value.shape[-2:] == (1, 1):
                    value = value.squeeze(-1).squeeze(-1)
                elif value.ndim != 2:
                    raise ValueError(f"VAE attention weight is not linear or 1x1 convolution: {external}")
            else:
                if value.ndim != 2:
                    raise ValueError(f"Native vae attention weight must be rank two: {native}")
                value = value.unsqueeze(-1).unsqueeze(-1)
        result[native if to_native else external] = value
    return result
