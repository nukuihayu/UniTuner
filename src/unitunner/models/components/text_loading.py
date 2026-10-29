"""Load local floating or packed text weights before optional layerwise quantization."""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from unitunner.artifacts.formats.quantization import FORMATS, canonical_quantization
from unitunner.artifacts.quantization import read_weight, tensor_json
from unitunner.artifacts.quantized_loading import load_quantized_model
from unitunner.config.text import TextEncodingConfig
from unitunner.models.components.weight_quantization import quantize_model
from unitunner.precision import FLOAT_DTYPES


def qwen_vl_base_key(key: str) -> str:
    """Map explicit Qwen/Comfy language and vision namespaces to base-model keys.

    Comfy files combine model.layers/embed_tokens/norm with model.visual;
    older Qwen files use the same language prefix and unprefixed visual keys.
    Check vision first so it never becomes language_model.visual. Callers keep
    their own inventory, duplicate, shape and source-format validation.
    """
    if key.startswith("model.visual."):
        return key.removeprefix("model.")
    if key.startswith("model."):
        return "language_model." + key.removeprefix("model.")
    return key


def text_weight_path(directory: Path) -> Path:
    """Resolve a local model file or the first standard shard without Hub lookup."""
    if directory.is_file():
        return directory
    if not directory.is_dir():
        raise FileNotFoundError(f"Text encoder does not exist: {directory}")
    path = directory / "model.safetensors"
    if path.is_file():
        return path
    shards = sorted(directory.glob("model-00001-of-*.safetensors"))
    if len(shards) != 1:
        raise ValueError("Text weights require model.safetensors or one standard first shard")
    return shards[0]


def text_dtype(config: TextEncodingConfig, default: torch.dtype) -> torch.dtype:
    """Resolve floating computation separately from packed weight storage."""
    if config.precision is None:
        return default
    return FLOAT_DTYPES[config.precision]


def load_text_state(
    model: nn.Module,
    tensors: Mapping[str, Tensor],
    *,
    dtype: torch.dtype,
    label: str,
    expected: Mapping[str, Tensor] | None = None,
    selected_prefix: str = "",
) -> None:
    """Validate full source inventory, including discarded branches, and assign on CPU.

    Namespace normalization and declared tied aliases belong to each encoder.
    Packed shapes are logical shapes, never byte widths. Only complete canonical
    calibration-free layouts are accepted; unknown sidecars fail strict loading.
    """
    topology = model.state_dict() if expected is None else expected
    try:
        sidecars: dict[str, str] = {}
        shapes: dict[str, torch.Size] = {}
        for key, value in tensors.items():
            if not key.endswith(".comfy_quant"):
                continue
            prefix = key.removesuffix("comfy_quant")
            config = tensor_json(value)
            format_name = config.get("format")
            spec = FORMATS.get(format_name) if isinstance(format_name, str) else None
            if spec is None or spec.calibrated:
                raise ValueError(f"{key}: requires a calibration-free weight layout")
            weight = read_weight(prefix + "weight", tensors.__getitem__, config, device=torch.device("cpu"))
            if not torch.isfinite(weight._qdata.float()).all():
                raise ValueError(f"{key}: nonfinite packed weight")
            shapes[prefix + "weight"] = weight.shape
            for name in (*weight.state_dict(prefix + "weight"), key):
                sidecars[name] = prefix + "weight"
        logical = (set(tensors) - sidecars.keys()) | shapes.keys()
        if logical != set(topology):
            raise ValueError(
                f"keys differ: missing={sorted(set(topology) - logical)}, extra={sorted(logical - set(topology))}"
            )
        for key in logical:
            if key in shapes:
                if shapes[key] != topology[key].shape:
                    raise ValueError(f"packed logical shape differs: {key}")
            else:
                value = tensors[key]
                if (
                    value.shape != topology[key].shape
                    or not value.is_floating_point()
                    or not torch.isfinite(value).all()
                    or not torch.isfinite(value.to(dtype)).all()
                ):
                    raise ValueError(f"invalid tensor shape/type; values invalid: {key}")
        selected = {selected_prefix + key for key in model.state_dict()}
        state = {
            key.removeprefix(selected_prefix): value
            for key, value in tensors.items()
            if sidecars.get(key, key) in selected
        }
        if not load_quantized_model(model, state, device=torch.device("cpu"), dtype=dtype):
            model.load_state_dict(
                {key: value.to(device="cpu", dtype=dtype) for key, value in state.items()}, strict=True, assign=True
            )
        _adapt_text_modules(model)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} {error}") from error


def _adapt_text_modules(model: nn.Module) -> None:
    from unitunner.models.components.quantized import QuantizedLinear
    from unitunner.models.flux.text.layers import T5TextFeedForward

    for name, layer in tuple(model.named_modules()):
        if type(layer).__name__ in ("T5DenseActDense", "T5DenseGatedActDense") and isinstance(
            getattr(layer, "wo", None), QuantizedLinear
        ):
            model.set_submodule(name, T5TextFeedForward(layer))


def prepare_text_model(model: nn.Module, config: TextEncodingConfig, *, family: str, device: torch.device) -> None:
    """Preserve packed source layers, optionally quantize dense ones, then place the encoder."""
    from unitunner.models.components.quantized import QuantizedLinear

    precision = canonical_quantization(config.quantization)
    for name, layer in model.named_modules():
        if not isinstance(layer, QuantizedLinear):
            continue
        if precision != "none":
            try:
                layer.validate_quantization(
                    precision,
                    rotation_size=config.quantization_rotation_size,
                    group_size=config.quantization_group_size,
                )
            except ValueError as error:
                raise ValueError(f"{name}: {error}") from error
    model.eval().requires_grad_(False)
    if config.quantization != "none":
        quantize_model(
            model,
            precision,
            family=family,
            rotation_size=config.quantization_rotation_size,
            group_size=config.quantization_group_size,
            device=device,
        )
    _adapt_text_modules(model)
    nn.Module.to(model, device=device).eval().requires_grad_(False)


def validate_cache_text(model: nn.Module) -> None:
    """Require frozen floating computations while accepting layout-specific packed dtypes."""
    from unitunner.models.components.quantized import QuantizedLinear

    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Cache text encoders must be frozen and in eval mode")
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            if module.compute_dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError("Cache text requires floating computation")
            continue
        if any(p.dtype not in (torch.float32, torch.float16, torch.bfloat16) for p in module.parameters(recurse=False)):
            raise ValueError("Cache text has unsupported floating parameters")


def text_execution(model: nn.Module) -> dict:
    """Record effective computation and packed layouts in cache execution identity."""
    from unitunner.models.components.quantized import QuantizedLinear

    execution = {
        "floating_dtypes": sorted(
            {
                str(p.dtype)
                for m in model.modules()
                if not isinstance(m, QuantizedLinear)
                for p in m.parameters(recurse=False)
            }
        ),
        "quantized": {
            name: {"layout": layer.config, "compute_dtype": str(layer.compute_dtype)}
            for name, layer in model.named_modules()
            if isinstance(layer, QuantizedLinear)
        },
    }

    return {} if execution["floating_dtypes"] == ["torch.float32"] and not execution["quantized"] else execution
