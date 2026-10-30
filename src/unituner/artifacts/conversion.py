"""Stream model-independent precision conversion into canonical Kitchen artifacts."""

import json
from contextlib import ExitStack, nullcontext
from fnmatch import fnmatchcase
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

import torch

from unituner.artifacts.formats.precision_policy import language_boundaries, protection_reason, resolve_family
from unituner.artifacts.formats.quantization import DEFAULT_GROUP_SIZE, DEFAULT_ROTATION_SIZE, FORMATS, PRECISIONS
from unituner.artifacts.quantization import (
    DTYPES,
    json_tensor,
    kitchen_types,
    quantize_weight,
    read_weight,
    serialize_weight,
    tensor_json,
)
from unituner.artifacts.tensors import TensorReader, resolve_shards, save_tensor_stream

_SUFFIXES = (
    "_scale",
    "_scale_2",
    "_s_rel",
    "_s_channel",
    "_correction",
    "_codebook",
    "_zeros",
    "_proj_down",
    "_proj_up",
    "_smooth_factor",
)


class ConversionReport(TypedDict):
    precision: str
    family: str
    converted: int
    preserved_quantized: int
    skipped: dict[str, str]


@torch.no_grad()
def convert_weights(
    source: Path,
    destination: Path,
    *,
    precision: str,
    device: str = "cpu",
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    rotation_size: int = DEFAULT_ROTATION_SIZE,
    group_size: int = DEFAULT_GROUP_SIZE,
    layout_spec: Path | None = None,
    on_incompatible: str = "error",
    backend: str | None = None,
    family: str = "auto",
    output_format: str = "source",
) -> ConversionReport:
    """Convert float or already-quantized shards without instantiating a model.

    Quantization selects nonempty 2D *.weight tensors. Float casts apply to
    all selected floating tensors. Excluded quantized layers keep every sidecar.
    AWQ/SVDQuant import accepts canonical calibrated tensors plus a JSON layer
    specification. Input smoothing is folded into weights when dequantizing or
    requantizing. This exports weights, not a quantized training installation.
    """
    if output_format not in ("source", "comfyui"):
        raise ValueError("Output format must be source or comfyui")
    if precision not in PRECISIONS or on_incompatible not in ("error", "keep"):
        raise ValueError("Unsupported precision or on-incompatible policy")
    if type(rotation_size) is not int or rotation_size < 1 or rotation_size & (rotation_size - 1):
        raise ValueError("rotation-size must be a positive power of two")
    if precision in ("int8_convrot", "convrot_w4a4", "convrot_w4a8", "asym_w4a8_int8", "w6a8_int8") and (
        rotation_size < 4 or (rotation_size.bit_length() - 1) % 2
    ):
        raise ValueError("ConvRot rotation-size must be a power of four (4, 16, 64, 256, ...)")
    if type(group_size) is not int or group_size < 4 or (16 % group_size and group_size % 16):
        raise ValueError("group-size must be >=4 and divide 16 or be a multiple of 16")
    selected_device = torch.device(device)
    if selected_device.type not in ("cpu", "cuda"):
        raise ValueError("Weight conversion device must be cpu or cuda")
    shards = resolve_shards(source)
    protected = (*shards, *((layout_spec,) if layout_spec is not None else ()))
    if destination.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Conversion destination must differ from source shards and layout specification")
    if output_format == "comfyui":
        from unituner.artifacts.comfyui_weights import comfyui_source

        with comfyui_source(source, family=family, precision=precision) as (staged, resolved):
            return convert_weights(
                staged,
                destination,
                precision=precision,
                device=device,
                include=include,
                exclude=exclude,
                rotation_size=rotation_size,
                group_size=group_size,
                layout_spec=layout_spec,
                on_incompatible=on_incompatible,
                backend=backend,
                family=resolved,
                output_format="source",
            )
    explicit = {} if layout_spec is None else json.loads(layout_spec.read_text(encoding="utf-8"))
    if not isinstance(explicit, dict) or any(
        not isinstance(k, str) or not isinstance(v, dict) for k, v in explicit.items()
    ):
        raise ValueError("Layout specification must map module prefixes to layer objects")
    report: ConversionReport = {
        "precision": precision,
        "family": family,
        "converted": 0,
        "preserved_quantized": 0,
        "skipped": {},
    }

    def selected(key):
        return (not include or any(fnmatchcase(key, pattern) for pattern in include)) and not any(
            fnmatchcase(key, pattern) for pattern in exclude
        )

    with ExitStack() as stack:
        readers, metadata = {}, None
        for shard in shards:
            reader = stack.enter_context(TensorReader(shard))
            current = reader.metadata()
            if metadata is None:
                metadata = current
            elif current != metadata:
                raise ValueError("Source shards must have identical metadata")
            for key in reader.keys():
                if key in readers:
                    raise ValueError(f"Duplicate tensor key across shards: {key}")
                readers[key] = reader
        resolved_family = resolve_family(family, metadata or {}, readers if precision in FORMATS else ())
        report["family"] = resolved_family
        boundaries = language_boundaries(readers)
        configs = {}
        for key, reader in readers.items():
            if key == "comfy_quant" or key.endswith(".comfy_quant"):
                configs[key.removesuffix("comfy_quant") + "weight"] = tensor_json(reader.read(key))
        legacy = json.loads((metadata or {}).get("_quantization_metadata", "{}"))
        for name, config in legacy.get("layers", {}).items():
            key = name + ".weight"
            if key in configs and configs[key] != config:
                raise ValueError(f"Conflicting quantization descriptions: {name}")
            configs[key] = config
        for prefix, config in explicit.items():
            key = prefix + ".weight" if prefix else "weight"
            configs[key] = {**configs.get(key, {}), **config}
        if any(key not in readers for key in configs):
            raise ValueError("Quantization metadata refers to a missing weight tensor")
        auxiliary = {key + suffix for key in configs for suffix in _SUFFIXES}
        for key in configs:
            prefix = key.removesuffix("weight")
            auxiliary.update(prefix + field for field in ("comfy_quant", "input_scale", "pre_quant_scale"))
        needs_kitchen = bool(configs) or precision in FORMATS
        if needs_kitchen:
            kitchen_types()
        context = import_module("comfy_kitchen").use_backend(backend) if backend and needs_kitchen else nullcontext()
        if backend and not needs_kitchen:
            raise ValueError("--backend applies to Kitchen layouts")
        metadata = dict(metadata or {})
        metadata.pop("_quantization_metadata", None)
        metadata["unituner.conversion"] = json.dumps(
            {
                "schema": 1,
                "precision": precision,
                "family": resolved_family,
                "include": include,
                "exclude": exclude,
                "kitchen": version("comfy-kitchen") if needs_kitchen else None,
            },
            separators=(",", ":"),
        )

        def read(key):
            return readers[key].read(key)

        def stream():
            for key in sorted(readers):
                if key in auxiliary:
                    continue
                config = configs.get(key)
                keep = precision == "keep" or not selected(key)
                if precision in FORMATS and selected(key) and (key == "weight" or key.endswith(".weight")):
                    reason = protection_reason(resolved_family, key, boundaries)
                    if reason is not None:
                        keep = True
                        report["skipped"][key] = reason
                if config and precision in FORMATS and FORMATS[precision].calibrated:
                    keep = keep or config.get("format") == FORMATS[precision].wire_format
                if config is not None:
                    try:
                        weight = read_weight(key, read, config, device=selected_device)
                    except (KeyError, TypeError, RuntimeError, ValueError) as error:
                        raise ValueError(f"Cannot load {key}: {error}") from error
                    if keep:
                        packed = serialize_weight(key, weight, config["format"])
                        conf_key = key.removesuffix("weight") + "comfy_quant"
                        packed[conf_key] = json_tensor({**config, **tensor_json(packed[conf_key])})
                        yield from packed.items()
                        for field in ("input_scale", "pre_quant_scale"):
                            extra = key.removesuffix("weight") + field
                            if extra in readers:
                                yield extra, read(extra)
                        report["preserved_quantized"] += 1
                        del weight, packed
                        continue
                    if precision in FORMATS and FORMATS[precision].calibrated:
                        raise ValueError(f"{precision} requires calibrated import with --precision keep")
                    try:
                        values = weight.dequantize()
                    except (RuntimeError, ValueError) as error:
                        raise ValueError(f"Cannot dequantize {key}: {error}") from error
                    del weight
                    smooth_key = key.removesuffix("weight") + "pre_quant_scale"
                    if smooth_key in readers:
                        smooth = read(smooth_key).to(device=selected_device, dtype=torch.float32)
                        if smooth.numel() != values.shape[-1]:
                            raise ValueError(f"{key}: pre_quant_scale must have one value per input feature")
                        values = values.float() * smooth.reshape(1, -1)
                else:
                    values = read(key)
                    if not keep and not values.is_floating_point() and (key == "weight" or key.endswith(".weight")):
                        raise ValueError(f"{key}: integer weight requires comfy_quant metadata or --layout-spec")
                if keep or not values.is_floating_point():
                    yield key, values
                    continue
                if not torch.isfinite(values.float()).all():
                    raise ValueError(f"Non-finite values in {key}")
                if precision in DTYPES:
                    cast = values.to(dtype=DTYPES[precision])
                    if not torch.isfinite(cast.float()).all():
                        raise ValueError(f"{key} overflows {precision}")
                    yield key, cast
                    report["converted"] += 1
                elif values.ndim == 2 and values.numel() and (key == "weight" or key.endswith(".weight")):
                    values = values.to(device=selected_device)
                    if values.dtype not in DTYPES.values():
                        values = values.float()
                    try:
                        weight = quantize_weight(values, precision, rotation_size=rotation_size, group_size=group_size)
                    except (ValueError, NotImplementedError) as error:
                        if on_incompatible == "error" or FORMATS[precision].calibrated:
                            raise ValueError(f"Cannot quantize {key}: {error}") from error
                        report["skipped"][key] = str(error)
                        yield key, values
                        continue
                    except RuntimeError as error:
                        raise ValueError(f"Cannot quantize {key}: {error}") from error
                    yield from serialize_weight(key, weight, precision).items()
                    report["converted"] += 1
                    del weight
                else:
                    yield key, values
                del values

        with context, torch.autocast(selected_device.type, enabled=False):
            save_tensor_stream(stream(), destination, metadata=metadata)
    return report
