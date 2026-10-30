"""Translate published Diffusers and ComfyUI transformer layouts to owned models.

Mappings follow the upstream Diffusers single-file converters and ComfyUI
loaders. Conversion never imports either runtime or discards unknown weights.
"""

import json
import re
from collections.abc import Iterable, Mapping

import torch
from torch import Tensor


def dequantize_transformer(tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Decode packed projections when a different layout requires split/fusion."""
    from unitunner.artifacts.quantization import read_weight, tensor_json

    result = dict(tensors)
    for key in tuple(tensors):
        if not key.endswith(".comfy_quant"):
            continue
        prefix = key.removesuffix("comfy_quant")
        if any(prefix + name in result for name in ("input_scale", "pre_quant_scale")):
            raise ValueError("Training layout conversion requires weights without static input scaling")
        config = tensor_json(tensors[key])
        weight = read_weight(prefix + "weight", tensors.__getitem__, config, device=torch.device("cpu"))
        for suffix in weight.state_dict("weight"):
            result.pop(prefix + suffix, None)
        result.pop(key)
        result[prefix + "weight"] = weight.dequantize()
    return result


def strip_transformer_prefix(tensors: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Remove one uniform component prefix; mixed or extra components stay invalid."""
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if tensors and all(key.startswith(prefix) for key in tensors):
            return {key[len(prefix) :]: value for key, value in tensors.items()}
    return dict(tensors)


def is_diffusers_transformer(family: str, keys: Iterable[str]) -> bool:
    """Identify layouts whose keys differ from the corresponding owned model."""
    prefixes = {
        "flux": ("x_embedder.", "context_embedder."),
        "flux2": ("x_embedder.", "context_embedder."),
        "flux3": ("x_embedders.", "context_embedder."),
        "wan": ("condition_embedder.",),
        "krea2": ("img_in.",),
        "minimax_h3": ("context_embedder.",),
    }.get(family, ())
    return any(key.startswith(prefixes) for key in keys)


def _replace(key: str, mapping: Mapping[str, str]) -> str:
    for old, new in mapping.items():
        key = key.replace(old, new)
    return key


def _flux_sources(key: str, *, second: bool) -> tuple[str, ...]:
    """Map native FLUX/FLUX.2 parameters to Diffusers, preserving Q/K/V order."""
    match = re.fullmatch(r"double_blocks\.(\d+)\.(img|txt)_attn\.qkv\.(weight|bias)", key)
    if match:
        index, stream, suffix = match.groups()
        names = ("to_q", "to_k", "to_v") if stream == "img" else ("add_q_proj", "add_k_proj", "add_v_proj")
        return tuple(f"transformer_blocks.{index}.attn.{name}.{suffix}" for name in names)
    match = re.fullmatch(r"single_blocks\.(\d+)\.linear1\.(weight|bias)", key)
    if match and not second:
        index, suffix = match.groups()
        root = f"single_transformer_blocks.{index}."
        return tuple(root + name + "." + suffix for name in ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"))
    if key.startswith("double_blocks."):
        key = _replace(
            key,
            {
                "double_blocks.": "transformer_blocks.",
                ".img_mod.lin.": ".norm1.linear.",
                ".txt_mod.lin.": ".norm1_context.linear.",
                ".img_attn.proj.": ".attn.to_out.0.",
                ".txt_attn.proj.": ".attn.to_add_out.",
                ".img_attn.norm.query_norm.scale": ".attn.norm_q.weight",
                ".img_attn.norm.key_norm.scale": ".attn.norm_k.weight",
                ".txt_attn.norm.query_norm.scale": ".attn.norm_added_q.weight",
                ".txt_attn.norm.key_norm.scale": ".attn.norm_added_k.weight",
                ".img_mlp.0.": ".ff.linear_in." if second else ".ff.net.0.proj.",
                ".img_mlp.2.": ".ff.linear_out." if second else ".ff.net.2.",
                ".txt_mlp.0.": ".ff_context.linear_in." if second else ".ff_context.net.0.proj.",
                ".txt_mlp.2.": ".ff_context.linear_out." if second else ".ff_context.net.2.",
            },
        )
    elif key.startswith("single_blocks."):
        key = _replace(
            key,
            {
                "single_blocks.": "single_transformer_blocks.",
                ".linear1.": ".attn.to_qkv_mlp_proj.",
                ".linear2.": ".attn.to_out." if second else ".proj_out.",
                ".modulation.lin.": ".norm.linear.",
                ".norm.query_norm.scale": ".attn.norm_q.weight",
                ".norm.key_norm.scale": ".attn.norm_k.weight",
            },
        )
    else:
        time = "time_guidance_embed" if second else "time_text_embed"
        key = _replace(
            key,
            {
                "img_in.": "x_embedder.",
                "txt_in.": "context_embedder.",
                "time_in.in_layer.": f"{time}.timestep_embedder.linear_1.",
                "time_in.out_layer.": f"{time}.timestep_embedder.linear_2.",
                "guidance_in.in_layer.": f"{time}.guidance_embedder.linear_1.",
                "guidance_in.out_layer.": f"{time}.guidance_embedder.linear_2.",
                "vector_in.in_layer.": f"{time}.text_embedder.linear_1.",
                "vector_in.out_layer.": f"{time}.text_embedder.linear_2.",
                "final_layer.adaLN_modulation.1.": "norm_out.linear.",
                "final_layer.linear.": "proj_out.",
                "double_stream_modulation_img.lin.": "double_stream_modulation_img.linear.",
                "double_stream_modulation_txt.lin.": "double_stream_modulation_txt.linear.",
                "single_stream_modulation.lin.": "single_stream_modulation.linear.",
            },
        )
    return (key,)


def _flux3_key(key: str) -> str:
    for old, new in {
        "emb_in.": "x_embedders.",
        "txt_in.": "context_embedder.",
        "content_mode_blocks.": "mode_blocks.",
        "single_blocks.": "single_transformer_blocks.",
        "time_in.in_layer.": "timestep_embedder.linear_1.",
        "time_in.out_layer.": "timestep_embedder.linear_2.",
        "ref_time_in.in_layer.": "target_timestep_embedder.linear_1.",
        "ref_time_in.out_layer.": "target_timestep_embedder.linear_2.",
        "final_layer.image.linear.": "proj_out.",
        "final_layer.image.adaLN_modulation.1.": "norm_out.linear.",
    }.items():
        if key.startswith(old):
            key = new + key[len(old) :]
            break
    return _replace(
        key,
        {
            ".lin.weight": ".linear.weight",
            ".q_proj.": ".attn.to_q.",
            ".k_proj.": ".attn.to_k.",
            ".v_proj.": ".attn.to_v.",
            ".attn_out.": ".attn.to_out.0.",
            ".norm.query_norm.scale": ".attn.norm_q.weight",
            ".norm.key_norm.scale": ".attn.norm_k.weight",
            ".mlp_in.": ".ff.linear_in.",
            ".mlp_out.": ".ff.linear_out.",
        },
    )


def _wan_key(key: str) -> str:
    # Norm2/3 exchange places in Diffusers; perform the swap simultaneously.
    key = key.replace(".norm2.", ".__norm2__.").replace(".norm3.", ".norm2.").replace(".__norm2__.", ".norm3.")
    return _replace(
        key,
        {
            "time_embedding.0.": "condition_embedder.time_embedder.linear_1.",
            "time_embedding.2.": "condition_embedder.time_embedder.linear_2.",
            "text_embedding.0.": "condition_embedder.text_embedder.linear_1.",
            "text_embedding.2.": "condition_embedder.text_embedder.linear_2.",
            "time_projection.1.": "condition_embedder.time_proj.",
            ".cross_attn.": ".attn2.",
            ".self_attn.": ".attn1.",
            ".o.": ".to_out.0.",
            ".q.": ".to_q.",
            ".k.": ".to_k.",
            ".v.": ".to_v.",
            ".k_img.": ".add_k_proj.",
            ".v_img.": ".add_v_proj.",
            ".norm_k_img.": ".norm_added_k.",
            "head.modulation": "scale_shift_table",
            "head.head.": "proj_out.",
            ".modulation": ".scale_shift_table",
            ".ffn.0.": ".ffn.net.0.proj.",
            ".ffn.2.": ".ffn.net.2.",
            "img_emb.proj.0.": "condition_embedder.image_embedder.norm1.",
            "img_emb.proj.1.": "condition_embedder.image_embedder.ff.net.0.proj.",
            "img_emb.proj.3.": "condition_embedder.image_embedder.ff.net.2.",
            "img_emb.proj.4.": "condition_embedder.image_embedder.norm2.",
        },
    )


def _krea_key(key: str) -> str:
    for old, new in {
        "first.": "img_in.",
        "tmlp.0.": "time_embed.linear_1.",
        "tmlp.2.": "time_embed.linear_2.",
        "tproj.1.": "time_mod_proj.",
        "txtmlp.0.scale": "txt_in.norm.weight",
        "txtmlp.1.": "txt_in.linear_1.",
        "txtmlp.3.": "txt_in.linear_2.",
        "txtfusion.": "text_fusion.",
        "blocks.": "transformer_blocks.",
        "last.linear.": "final_layer.linear.",
        "last.norm.scale": "final_layer.norm.weight",
        "last.modulation.lin": "final_layer.scale_shift_table",
    }.items():
        if key.startswith(old):
            key = new + key[len(old) :]
            break
    return _replace(
        key,
        {
            ".attn.wq.": ".attn.to_q.",
            ".attn.wk.": ".attn.to_k.",
            ".attn.wv.": ".attn.to_v.",
            ".attn.wo.": ".attn.to_out.0.",
            ".attn.gate.": ".attn.to_gate.",
            ".attn.qknorm.qnorm.scale": ".attn.norm_q.weight",
            ".attn.qknorm.knorm.scale": ".attn.norm_k.weight",
            ".mlp.": ".ff.",
            ".prenorm.scale": ".norm1.weight",
            ".postnorm.scale": ".norm2.weight",
            ".mod.lin": ".scale_shift_table",
        },
    )


def _h3_sources(key: str) -> tuple[str, ...]:
    key = _replace(
        key,
        {
            "video_patch_proj.": "proj_in.",
            "audio_patch_proj.": "audio_proj_in.",
            "condition_proj.": "context_embedder.",
            "time_embedder.proj_in.": "time_embedder.linear_1.",
            "time_embedder.proj_out.": "time_embedder.linear_2.",
            "token_refiner.blocks.": "token_refiner.refiner_blocks.",
            "final_layer.norm.": "norm_out.norm.",
            "final_layer.adaln_proj.linear.": "norm_out.linear.",
            "final_layer.video_out.": "proj_out.",
            "final_layer.audio_out.": "audio_proj_out.",
        },
    )
    if key.startswith("blocks."):
        key = "transformer_blocks." + key[len("blocks.") :]
    if key.endswith(".attn.qkv_proj.weight"):
        return tuple(key.replace(".qkv_proj.", f".to_{part}.") for part in ("q", "k", "v"))
    return (
        _replace(
            key,
            {
                ".attn.q_norm.": ".attn.norm_q.",
                ".attn.k_norm.": ".attn.norm_k.",
                ".attn.out_proj.": ".attn.to_out.0.",
                ".mlp.fc1.": ".ff.net.0.proj.",
                ".mlp.fc2.": ".ff.net.2.",
            },
        ),
    )


def official_projection_sources(family: str, key: str) -> tuple[str, ...]:
    """Map a native parameter to ordered official parameters, retaining fusion boundaries."""
    if family in {"flux", "flux2"}:
        return _flux_sources(key, second=family == "flux2")
    if family == "flux3":
        return (_flux3_key(key),)
    if family == "wan":
        return (_wan_key(key),)
    if family == "krea2":
        return (_krea_key(key),)
    if family == "minimax-h3":
        return _h3_sources(key)
    return (key,)


def convert_transformer(
    family: str, tensors: Mapping[str, Tensor], expected: Mapping[str, Tensor], *, external: bool | None = None
) -> dict[str, Tensor]:
    """Convert only recognized external layouts, retaining strict inventory checks.

    Native quantized layouts pass through unchanged. Fused external quantized
    projections cannot be split as floating tensors and fail explicitly.
    """
    source = strip_transformer_prefix(tensors)
    if family == "z_image":
        if any(".attention.qkv." in key for key in source):
            source = dequantize_transformer(source)
        converted: dict[str, Tensor] = {}
        for key, value in source.items():
            for old, new in (("final_layer.", "all_final_layer.2-1."), ("x_embedder.", "all_x_embedder.2-1.")):
                if key.startswith(old):
                    key = new + key[len(old) :]
                    break
            key = _replace(
                key,
                {
                    ".attention.out.": ".attention.to_out.0.",
                    ".attention.q_norm.": ".attention.norm_q.",
                    ".attention.k_norm.": ".attention.norm_k.",
                },
            )
            parts = [(key, value)]
            if key.endswith(".attention.qkv.weight"):
                keys = [key.replace(".qkv.", f".to_{part}.") for part in ("q", "k", "v")]
                if not value.is_floating_point() or any(k not in expected for k in keys):
                    raise ValueError(f"Unsupported external fused Z-Image projection: {key}")
                widths = [expected[k].shape[0] for k in keys]
                parts = list(zip(keys, value.split(widths, dim=0), strict=True))
            if key == "norm_final.weight":
                # Legacy original checkpoint includes this unused affine norm.
                width = expected["cap_embedder.1.weight"].shape[0]
                if value.shape != (width,) or not value.is_floating_point() or not torch.isfinite(value).all():
                    raise ValueError("Invalid legacy Z-Image final norm")
                continue
            for target, tensor in parts:
                if target in converted:
                    raise ValueError(f"Duplicate converted transformer tensor: {target}")
                converted[target] = tensor
        return converted
    if family == "qwen_image_2_1":
        for key in tuple(source):
            if re.fullmatch(r"transformer_blocks\.\d+\.attn\.comfy_attention\.config", key):
                value = source.pop(key)
                projection = key.removesuffix("comfy_attention.config") + "to_q.weight"
                if projection not in source or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() > 65536:
                    raise ValueError(f"Invalid ComfyUI attention configuration: {key}")
                try:
                    settings = json.loads(value.cpu().numpy().tobytes())
                except (ValueError, UnicodeDecodeError) as error:
                    raise ValueError(f"Invalid ComfyUI attention configuration: {key}") from error
                choices = settings if isinstance(settings, list) else [settings]
                if not choices or any(
                    not isinstance(item, dict) or not isinstance(item.get("attention"), str) for item in choices
                ):
                    raise ValueError(f"Invalid ComfyUI attention configuration: {key}")
                # Backend preferences (including quantized inference attention)
                # carry no model weights. Training owns its attention backend.
                continue
            if key.endswith(".img_mlp.gate_up.weight"):
                prefix = key.removesuffix("weight")
                if prefix + "comfy_quant" in source:
                    # Only the fused projection changes layout. Keep unrelated
                    # attention/MLP weights in their original packed storage.
                    local = {name: value for name, value in source.items() if name.startswith(prefix)}
                    decoded = dequantize_transformer(local)
                    for name in local:
                        del source[name]
                    source.update(decoded)
                value = source.pop(key)
                if value.ndim != 2 or value.shape[0] % 2 or not value.is_floating_point():
                    raise ValueError(f"Invalid fused Qwen Image 2.1 MLP: {key}")
                for name, chunk in zip(("gate_layer", "proj"), value.chunk(2, dim=0), strict=True):
                    target = key.replace("gate_up", name)
                    if any(existing.startswith(target.removesuffix("weight")) for existing in source):
                        raise ValueError(f"Duplicate converted transformer tensor: {target}")
                    if target in expected and chunk.shape != expected[target].shape:
                        raise ValueError(f"Invalid fused Qwen Image 2.1 MLP shape: {key}")
                    source[target] = chunk
        return source
    if family not in {"flux", "flux2", "flux3", "wan", "krea2", "minimax_h3"}:
        return source
    # Common native keys alone are not a format signal (e.g. Wan patch_embedding).
    if external is None:
        external = is_diffusers_transformer(family, source)
    if not external:
        return source
    source = dequantize_transformer(source)
    remaining = dict(source)
    converted = {}
    for native, template in expected.items():
        if family in {"flux", "flux2"}:
            keys = _flux_sources(native, second=family == "flux2")
        elif family == "minimax_h3":
            keys = _h3_sources(native)
        else:
            keys = (({"flux3": _flux3_key, "wan": _wan_key, "krea2": _krea_key}[family])(native),)
        if any(key not in remaining for key in keys):
            continue
        parts = [remaining.pop(key) for key in keys]
        if len(parts) > 1 and any(not part.is_floating_point() for part in parts):
            raise ValueError(f"External fused projection requires floating weights: {native}")
        value = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        if family == "minimax_h3" and native.endswith(".mlp.fc1.weight"):
            # Diffusers SwiGLU stores value rows before gate; native stores gate first.
            if value.shape[0] % 2:
                raise ValueError(f"Invalid MiniMaxH3 SwiGLU rows: {keys[0]}")
            values, gate = value.chunk(2, dim=0)
            value = torch.cat((gate, values), dim=0)
        if keys[0].startswith("norm_out.linear.") and family in {"flux", "flux2", "flux3"}:
            if value.shape[0] % 2:
                raise ValueError(f"Invalid scale/shift rows: {keys[0]}")
            scale, shift = value.chunk(2, dim=0)
            value = torch.cat((shift, scale), dim=0)
        if family == "krea2" and native.startswith("blocks.") and native.endswith(".mod.lin"):
            if value.numel() != template.numel():
                raise ValueError(f"Invalid Krea2 modulation shape: {keys[0]}")
            value = value.reshape(template.shape)
        converted[native] = value
    duplicates = converted.keys() & remaining.keys()
    if duplicates:
        raise ValueError(f"Duplicate converted transformer tensors: {sorted(duplicates)}")
    if (
        family == "minimax_h3"
        and any(k.startswith("context_embedder.") for k in source)
        and "rope.inv_freq" in expected
        and "rope.inv_freq" not in converted
        and "rope.inv_freq" not in remaining
    ):
        length = expected["rope.inv_freq"].numel()
        converted["rope.inv_freq"] = 10000.0 ** (-torch.arange(length, dtype=torch.float32) / length)
    return {**remaining, **converted}
