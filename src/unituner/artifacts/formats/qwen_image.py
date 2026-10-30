# Copyright 2025 The Qwen-Image Team, Wan Team and The HuggingFace Team.
# Apache-2.0; see THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""Convert explicitly selected Qwen Image artifact namespaces without model inference."""

import re
from collections.abc import Mapping, Sequence

from torch import Tensor


def _residual_key(key: str) -> str:
    """Translate native residual fields while preserving their tensor layouts."""
    for native, external in (
        ("norm1", "residual.0"),
        ("conv1", "residual.2"),
        ("norm2", "residual.3"),
        ("conv2", "residual.6"),
        ("conv_shortcut", "shortcut"),
    ):
        key = key.replace(f".{native}.", f".{external}.")
    return key


def vae_key_map(native_keys: Sequence[str], *, num_blocks: int, layers_per_block: int) -> dict[str, str]:
    """Map complete native Qwen VAE keys bijectively to ComfyUI Wan VAE keys.

    Encoder entries retain flat residual/attention/resample indices. Decoder
    stages flatten layers_per_block+1 residuals followed by one upsample, except
    the final stage. Unlike a fixed published table, this preserves actual
    reduced topology. No weight reshaping or value conversion occurs.
    """
    if num_blocks < 1 or layers_per_block < 1:
        raise ValueError("Qwen vae conversion requires positive stage and residual counts")
    result: dict[str, str] = {}
    roots = {
        "quant_conv": "conv1",
        "post_quant_conv": "conv2",
        "encoder.conv_in": "encoder.conv1",
        "encoder.norm_out": "encoder.head.0",
        "encoder.conv_out": "encoder.head.2",
        "decoder.conv_in": "decoder.conv1",
        "decoder.norm_out": "decoder.head.0",
        "decoder.conv_out": "decoder.head.2",
    }
    for key in native_keys:
        path, _, suffix = key.rpartition(".")
        if path in roots:
            external = roots[path] + "." + suffix
        elif match := re.fullmatch(r"(encoder|decoder)\.mid_block\.(resnets|attentions)\.(\d+)\.(.+)", key):
            side, kind, index, tail = match.groups()
            if kind == "attentions" and index == "0":
                external = f"{side}.middle.1.{tail}"
            elif kind == "resnets" and index in ("0", "1"):
                external = _residual_key(f"{side}.middle.{2 * int(index)}.{tail}")
            else:
                raise ValueError(f"Unsupported Qwen vae middle entry: {key}")
        elif match := re.fullmatch(r"encoder\.down_blocks\.(\d+)\.(.+)", key):
            external = _residual_key(f"encoder.downsamples.{match[1]}.{match[2]}")
        elif match := re.fullmatch(r"decoder\.up_blocks\.(\d+)\.(resnets|upsamplers)\.(\d+)\.(.+)", key):
            stage, kind, index, tail = match.groups()
            if int(stage) >= num_blocks:
                raise ValueError("Qwen vae decoder stage exceeds explicit topology")
            offset = int(stage) * (layers_per_block + 2)
            if kind == "resnets" and int(index) <= layers_per_block:
                external = _residual_key(f"decoder.upsamples.{offset + int(index)}.{tail}")
            elif kind == "upsamplers" and index == "0" and int(stage) < num_blocks - 1:
                external = f"decoder.upsamples.{offset + layers_per_block + 1}.{tail}"
            else:
                raise ValueError(f"Unsupported Qwen vae decoder entry: {key}")
        else:
            raise ValueError(f"Unsupported Qwen vae key: {key}")
        if key in result or external in result.values():
            raise ValueError("Qwen vae mapping must be bijective")
        result[key] = external
    return result


def convert_vae_tensors(
    tensors: Mapping[str, Tensor], keys: Mapping[str, str], *, to_native: bool
) -> dict[str, Tensor]:
    """Convert an exact Qwen VAE key inventory, sharing unchanged tensor storage.

    Native and ComfyUI use identical temporal/spatial weight and RMS gamma
    layouts. Missing/extra/colliding names fail; shape and finite-value validation
    belong to the strict native loader after conversion.
    """
    if len(keys) != len(set(keys.values())):
        raise ValueError("Qwen vae mapping must be bijective")
    expected = set(keys.values()) if to_native else set(keys)
    if set(tensors) != expected:
        raise ValueError(
            f"Qwen vae keys differ: missing={sorted(expected - tensors.keys())}, extra={sorted(tensors.keys() - expected)}"
        )
    return {
        native if to_native else external: tensors[external if to_native else native]
        for native, external in keys.items()
    }


def image21_vae_key_map(
    native_keys: Sequence[str], *, num_blocks: int, layers_per_block: int, is_residual: bool
) -> dict[str, str]:
    """Map Image 2.1's nested residual stages to released ComfyUI VAE keys."""
    if not is_residual:
        return vae_key_map(native_keys, num_blocks=num_blocks, layers_per_block=layers_per_block)
    result: dict[str, str] = {}
    for key in native_keys:
        if match := re.fullmatch(r"(encoder|decoder)\.(down_blocks|up_blocks)\.(\d+)\.resnets\.(\d+)\.(.+)", key):
            side, _, stage, index, tail = match.groups()
            direction = "downsamples" if side == "encoder" else "upsamples"
            external = _residual_key(f"{side}.{direction}.{stage}.{direction}.{index}.{tail}")
        elif match := re.fullmatch(
            r"(encoder|decoder)\.(down_blocks|up_blocks)\.(\d+)\.(downsampler|upsampler)\.(.+)", key
        ):
            side, _, stage, _, tail = match.groups()
            direction = "downsamples" if side == "encoder" else "upsamples"
            index = layers_per_block + (side == "decoder")
            external = f"{side}.{direction}.{stage}.{direction}.{index}.{tail}"
        else:
            external = vae_key_map((key,), num_blocks=num_blocks, layers_per_block=layers_per_block)[key]
        result[key] = external
    if len(result) != len(set(result.values())):
        raise ValueError("Qwen Image2.1 vae mapping must be bijective")
    return result


def remove_prefix(tensors: Mapping[str, Tensor], prefix: str) -> dict[str, Tensor]:
    """Remove one explicitly selected anchored prefix from every artifact key."""
    if any(not key.startswith(prefix) for key in tensors):
        raise ValueError("Qwen artifact keys must all use the selected prefix")
    return {key.removeprefix(prefix): value for key, value in tensors.items()}


def convert_transformer_tensors(tensors: Mapping[str, Tensor], *, prefix: str, variant: str) -> dict[str, Tensor]:
    """Normalize ComfyUI keys and validate its empty edit-2511 zero-time marker.

    Prefix selection is explicit and uniform; no mixed namespace inference.
    The marker is metadata, never a learned parameter. It is optional for edit-2511
    and forbidden for every other explicitly selected variant. Preserve inputs.
    """
    result = remove_prefix(tensors, prefix)
    marker = result.pop("__index_timestep_zero__", None)
    if marker is not None and (variant != "edit-2511" or marker.shape != (0,)):
        raise ValueError("Qwen empty index-timestep-zero marker requires edit-2511")
    return result
