"""Resolve native and published transformer topology and artifact tensors."""

import json
from pathlib import Path

from pydantic import BaseModel

from unitunner.artifacts.tensors import TensorReader, iter_tensors, resolve_shards, resolve_weight_path


def resolve_transformer_config[ConfigT: BaseModel](
    path: Path,
    cls: type[ConfigT],
    *,
    metadata_key: str,
    config: ConfigT | None,
    variant: str | None,
    version: str | None,
    mismatch_message: str,
) -> ConfigT:
    """Resolve explicit, embedded, then published topology before loading tensors.

    Explicit configuration must agree with embedded metadata and is authoritative
    over variant/version hints. Without it, hints select embedded sampling
    semantics or the published configuration using the same family rules.
    """
    serialized = transformer_metadata(path).get(metadata_key)
    embedded = None if serialized is None else cls.model_validate_json(serialized)
    if config is not None:
        if embedded is not None and config != embedded:
            raise ValueError(mismatch_message)
        return config
    if embedded is not None:
        return select_transformer_variant(embedded, variant=variant, version=version)
    return published_config(path, cls, variant=variant, version=version)


def select_transformer_variant[ConfigT: BaseModel](
    config: ConfigT, *, variant: str | None, version: str | None
) -> ConfigT:
    """Apply caller-owned sampling semantics while retaining checkpoint geometry.

    Turbo/base and Wan expert versions cannot always be distinguished by weights.
    FLUX.2 versions select different architectures and must still agree.
    """
    values = config.model_dump()
    if version is not None and "version" in values:
        if type(config).__name__ == "Flux2Config" and values["version"] != version:
            raise ValueError("FLUX.2 version differs from checkpoint topology")
        values["version"] = version
    if variant is not None and "variant" in values:
        values["variant"] = variant
    return type(config).model_validate_json(json.dumps(values))


def transformer_tensors(path: Path):
    """Read weights and translate ComfyUI's legacy quantization metadata."""
    from unitunner.artifacts.quantization import json_tensor

    tensors = dict(iter_tensors(path))
    metadata = transformer_metadata(path)
    serialized = metadata.get("_quantization_metadata")
    if serialized is not None:
        layers = json.loads(serialized)["layers"]
        for name, config in layers.items():
            key = name + ".comfy_quant"
            if key in tensors:
                raise ValueError(f"Duplicate ComfyUI quantization description: {name}")
            tensors[key] = json_tensor(config)
    return tensors


def transformer_metadata(path: Path) -> dict[str, str]:
    """Read consistent metadata across every shard, including indexed artifacts."""
    metadata: dict[str, str] = {}
    for shard in resolve_shards(path):
        with TensorReader(shard) as reader:
            for key, value in reader.metadata().items():
                if key in metadata and metadata[key] != value:
                    raise ValueError(f"Conflicting transformer shard metadata: {key}")
                metadata[key] = value
    return metadata


def published_config[ConfigT: BaseModel](
    path: Path, cls: type[ConfigT], *, variant: str | None = None, version: str | None = None
) -> ConfigT:
    """Use a local configuration, otherwise select the released architecture.

    Header dimensions distinguish published model sizes. Semantic variants such
    as Base/Turbo are supplied by the caller rather than guessed from filenames.
    A sidecar can describe a custom topology; complete key/shape validation still
    happens in the family loader before any model storage is materialized.
    """
    weight = resolve_weight_path(path)
    sidecars = (weight.with_suffix(".json"), weight.parent / "config.json")
    values: dict[str, object] = {}
    raw: dict[str, object] = {}
    for sidecar in sidecars:
        if sidecar.is_file() and not sidecar.name.endswith(".index.json"):
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"Transformer configuration must be an object: {sidecar}")
            break
    name = cls.__name__
    aliases = {
        "ZImageConfig": {
            "axes_lens": "axes_lengths",
            "t_scale": "time_scale",
            "n_heads": "num_heads",
            "n_kv_heads": "num_kv_heads",
            "n_layers": "num_layers",
            "n_refiner_layers": "num_refiner_layers",
            "cap_feat_dim": "caption_dim",
            "all_patch_size": "patch_sizes",
            "all_f_patch_size": "frame_patch_sizes",
        },
        "FluxConfig": {
            "num_layers": "depth",
            "num_single_layers": "depth_single_blocks",
            "joint_attention_dim": "context_in_dim",
            "pooled_projection_dim": "vec_in_dim",
            "num_attention_heads": "num_heads",
            "axes_dims_rope": "axes_dim",
            "guidance_embeds": "guidance_embed",
        },
        "Flux2Config": {
            "num_layers": "depth",
            "num_single_layers": "depth_single_blocks",
            "joint_attention_dim": "context_in_dim",
            "num_attention_heads": "num_heads",
            "axes_dims_rope": "axes_dim",
            "rope_theta": "theta",
        },
        "Flux3Config": {
            "num_layers": "depth",
            "num_single_layers": "depth_single",
            "joint_attention_dim": "text_channels",
            "in_channels": "image_channels",
            "num_attention_heads": "num_heads",
            "axes_dims_rope": "axes_dim",
            "rope_theta": "theta",
        },
        "WanConfig": {
            "model_type": "variant",
            "dim": "hidden_size",
            "num_attention_heads": "num_heads",
            "freq_dim": "timestep_dim",
            "text_len": "text_length",
            "eps": "norm_eps",
            "cross_attn_norm": "cross_attention_norm",
            "in_dim": "in_channels",
            "out_dim": "out_channels",
        },
        "Krea2Config": {
            "num_attention_heads": "num_heads",
            "num_key_value_heads": "num_kv_heads",
            "timestep_embed_dim": "timestep_dim",
            "text_hidden_dim": "text_dim",
            "num_text_layers": "text_layers",
            "text_num_attention_heads": "text_heads",
            "text_num_key_value_heads": "text_kv_heads",
            "axes_dims_rope": "rotary_axes",
            "rope_theta": "theta",
        },
        "MiniMaxH3Config": {
            "num_refiner_layers": "token_refiner_num_layers",
            "ffn_dim": "ffn_hidden_size",
            "freq_dim": "timestep_input_dim",
            "time_embed_hidden_dim": "time_embed_hidden_size",
            "rope_freq_dim": "rope_inv_freq_len",
        },
    }.get(name, {})
    fields = cls.model_fields
    for key, value in raw.items():
        target = aliases.get(key, key)
        if target in fields:
            if value is None and fields[target].default is not None:
                continue
            # JSON arrays correspond to the immutable native tuple dimensions.
            if isinstance(fields[target].default, tuple) and isinstance(value, list):
                value = tuple(value)
            values[target] = value
    if name in {"FluxConfig", "Flux2Config", "Flux3Config", "WanConfig"}:
        heads = raw.get("num_attention_heads")
        head_dim = raw.get("attention_head_dim")
        if isinstance(heads, int) and isinstance(head_dim, int):
            values["hidden_size"] = heads * head_dim
    if name == "WanConfig" and isinstance(values.get("qk_norm"), str):
        if values["qk_norm"] != "rms_norm_across_heads":
            raise ValueError("Wan requires rms_norm_across_heads QK normalization")
        values["qk_norm"] = True
    if name == "Krea2Config" and "_class_name" in raw:
        # Diffusers exposes packed patch channels; the native model owns packing.
        channels = raw.get("in_channels", 64)
        if not isinstance(channels, int) or channels % 4:
            raise ValueError("Krea2 packed channels must be divisible by four")
        values["in_channels"] = channels // 4
        heads = raw.get("num_attention_heads", 48)
        head_dim = raw.get("attention_head_dim", 128)
        if isinstance(heads, int) and isinstance(head_dim, int):
            values["hidden_size"] = heads * head_dim
    shapes: dict[str, tuple[int, ...]] = {}
    logical_shapes: dict[str, tuple[int, ...]] = {}
    for shard in resolve_shards(path):
        with TensorReader(shard) as reader:
            for key in reader.keys():
                original = key
                for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
                    if key.startswith(prefix):
                        key = key[len(prefix) :]
                        break
                # Only header dimensions; no weight payload or network access.
                shapes[key] = reader.shape(original)
                if name == "QwenImage21Config" and key.endswith(".comfy_quant"):
                    from unitunner.artifacts.quantization import tensor_json

                    # Packed storage can pad rows or pack several columns into a
                    # byte. Only this small JSON tensor is needed for topology.
                    quantization = tensor_json(reader.read(original))
                    shape = quantization.get("orig_shape")
                    if shape is not None:
                        if (
                            not isinstance(shape, list)
                            or len(shape) != 2
                            or any(type(n) is not int or n < 1 for n in shape)
                        ):
                            raise ValueError(f"Invalid Qwen Image 2.1 logical shape: {key}")
                        logical_shapes[key.removesuffix("comfy_quant") + "weight"] = tuple(shape)
    shapes.update(logical_shapes)
    if name == "Flux2Config":
        width = shapes.get("txt_in.weight", shapes.get("context_embedder.weight", (0, 0)))[1]
        detected = {7680: "klein-4b", 12288: "klein-9b", 15360: "dev"}.get(width, "dev")
        if version is not None and width in (7680, 12288, 15360) and version != detected:
            raise ValueError("FLUX.2 version differs from checkpoint dimensions")
        selected = version or str(values.get("version", detected))
        from unitunner.models.flux2.config.model import Flux2Config, flux2_config

        selection = Flux2Config.model_validate_json(
            json.dumps({"version": selected, "variant": variant or values.get("variant", "distilled")})
        )
        preset = flux2_config(selection.version, variant=selection.variant)
        values = {**preset.model_dump(), **values}
    elif name == "FluxConfig":
        shape = shapes.get("img_in.weight", shapes.get("x_embedder.weight", (3072, 64)))
        values.setdefault("in_channels", shape[1])
        values.setdefault("variant", "fill" if shape[1] == 384 else "dev")
        values.setdefault(
            "guidance_embed", any(k.startswith(("guidance_in.", "time_text_embed.guidance_embedder.")) for k in shapes)
        )
    elif name == "Flux3Config":
        values.setdefault(
            "variant",
            "distilled" if any(k.startswith(("ref_time_in.", "target_timestep_embedder.")) for k in shapes) else "base",
        )
    elif name == "WanConfig":
        shape = shapes.get("patch_embedding.weight")
        if shape is not None and shape[0] in (1536, 3072, 5120):
            size = shape[0]
            values.setdefault("hidden_size", size)
            values.setdefault("in_channels", shape[1])
            values.setdefault("patch_size", tuple(shape[2:]))
            values.setdefault("num_heads", size // 128)
            values.setdefault("num_layers", {1536: 30, 3072: 30, 5120: 40}[size])
            values.setdefault("ffn_dim", {1536: 8960, 3072: 14336, 5120: 13824}[size])
            if size == 3072:
                values.setdefault("out_channels", 48)
                values.setdefault("version", "2.2")
        image = any(k.startswith(("img_emb.", "condition_embedder.image_embedder.")) for k in shapes)
        values.setdefault("version", "2.1" if image else "2.2" if shape is not None and shape[1] != 16 else "2.1")
        values.setdefault("variant", "i2v" if image or (shape is not None and shape[1] == 36) else "t2v")
        values.setdefault("cross_attention_norm", any(".norm3." in k or ".norm2.weight" in k for k in shapes))
    elif name == "MiniMaxH3Config" and "adaln_t_table" in shapes:
        values.setdefault("adaln_curve_grid", shapes["adaln_t_table"][0])
        values.setdefault("time_embed_dim", shapes["adaln_t_table"][1])
    elif name == "MiniMaxH3Config" and "adaln_basis" in shapes:
        values.setdefault("time_embed_dim", shapes["adaln_basis"][0])
        values.setdefault("adaln_rank", shapes["adaln_basis"][1])
    elif name == "QwenImage21Config":
        required = (
            "img_in.weight",
            "proj_out.weight",
            "txt_in.text_norm.weight",
            "transformer_blocks.0.attn.norm_q.weight",
        )
        if any(key not in shapes for key in required):
            raise ValueError("Qwen Image 2.1 checkpoint is missing architecture tensors")
        image, output, text, norm = (shapes[key] for key in required)
        if len(image) != 2 or len(output) != 2 or len(text) != 1 or len(norm) != 1:
            raise ValueError("Invalid Qwen Image 2.1 architecture tensor dimensions")
        width, channels = image
        head_dim = norm[0]
        fused = "transformer_blocks.0.img_mlp.gate_up.weight" in shapes
        mlp = shapes.get(
            "transformer_blocks.0.img_mlp.gate_up.weight" if fused else "transformer_blocks.0.img_mlp.proj.weight"
        )
        factor = 2 if fused else 1
        if (
            min(width, channels, head_dim, *output, *text) < 1
            or width % head_dim
            or output[1] != width
            or mlp is None
            or len(mlp) != 2
            or mlp[1] != width
            or mlp[0] < factor * width
            or mlp[0] % (factor * width)
        ):
            raise ValueError("Inconsistent Qwen Image 2.1 attention or MLP dimensions")
        blocks = {
            int(key.split(".")[1])
            for key in shapes
            if key.startswith("transformer_blocks.") and key.split(".")[1].isdecimal()
        }
        if blocks != set(range(len(blocks))):
            raise ValueError("Qwen Image 2.1 transformer blocks must be contiguous from zero")
        detected = {
            "in_channels": channels,
            "out_channels": output[0],
            "context_in_dim": text[0],
            "attention_head_dim": head_dim,
            "num_attention_heads": width // head_dim,
            "num_layers": len(blocks),
            "mlp_ratio": mlp[0] // (factor * width),
        }
        for field, value in detected.items():
            if field in values and values[field] != value:
                raise ValueError(f"Qwen Image 2.1 configuration disagrees with checkpoint: {field}")
            values[field] = value
        # RoPE axis allocation and epsilon are not encoded in the weights:
        # retain the published defaults, or the caller's explicit sidecar.
    elif name == "QwenImageConfig":
        if "__index_timestep_zero__" in shapes or raw.get("zero_cond_t") is True:
            values.setdefault("variant", "edit-2511")
        elif "time_text_embed.addition_t_embedding.weight" in shapes or raw.get("use_additional_t_cond") is True:
            values.setdefault("variant", "layered")
    if variant is not None and "variant" in fields:
        values["variant"] = variant
    if version is not None and "version" in fields:
        values["version"] = version
    # Validate through JSON so tuple and strict scalar semantics match published config files.
    return cls.model_validate_json(json.dumps(values))
