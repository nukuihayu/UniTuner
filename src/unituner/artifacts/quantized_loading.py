"""Strict native loading of canonical quantized Linear weights and sidecars."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from unituner.artifacts.formats.quantization import FORMATS
from unituner.artifacts.quantization import read_weight, tensor_json
from unituner.models.components.quantized import QuantizedLinear


def load_quantized_model(
    model: nn.Module,
    tensors: Mapping[str, Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype | None,
    floating_dtypes: Mapping[str, torch.dtype] | None = None,
) -> bool:
    """Validate the entire artifact before replacing projections or assigning weights."""
    configs = {
        key.removesuffix("comfy_quant"): tensor_json(value)
        for key, value in tensors.items()
        if key == "comfy_quant" or key.endswith(".comfy_quant")
    }
    if not configs:
        return False
    expected = model.state_dict()
    quantized_keys = {prefix + "weight" for prefix in configs}
    dense_dtypes = {
        value.dtype
        for key, value in tensors.items()
        if key in expected and key not in quantized_keys and value.is_floating_point()
    }
    # A dtype-cast republication changes compute precision, not scale precision
    # or the logical dtype used to decode the original quantized representation.
    inferred_dtype = next(iter(dense_dtypes)) if len(dense_dtypes) == 1 else None
    owned: set[str] = set()
    replacements = []
    converted: dict[str, Tensor] = {}
    for prefix, config in configs.items():
        name = prefix.removesuffix(".")
        try:
            projection = model.get_submodule(name)
        except AttributeError as error:
            raise ValueError(f"Quantization metadata targets missing module {name}") from error
        if not name or not isinstance(projection, nn.Linear):
            raise ValueError(f"{name}: quantized weight must belong to a named Linear")
        format_name = config.get("format")
        spec = FORMATS.get(format_name) if isinstance(format_name, str) else None
        if spec is None:
            raise ValueError(f"{name}: training requires a supported Kitchen quantization format")
        key = prefix + "weight"
        weight = read_weight(key, tensors.__getitem__, config, device=device)
        if tuple(weight.shape) != (projection.out_features, projection.in_features):
            raise ValueError(f"{key}: logical shape differs from model topology")
        if prefix + "input_scale" in tensors or prefix + "pre_quant_scale" in tensors:
            raise ValueError(f"{key}: training does not accept static input scaling/smoothing sidecars")
        compute = dtype if dtype is not None else inferred_dtype or weight._params.orig_dtype
        if floating_dtypes is not None:
            compute = floating_dtypes.get(key, compute)
        local = {k: tensors[prefix + k].to(device) for k in weight.state_dict("weight")}
        local["comfy_quant"] = tensors[prefix + "comfy_quant"].to(device)
        owned.update(prefix + k for k in local)
        replacements.append((name, compute, local))
    base_keys = set(tensors) - owned | {prefix + "weight" for prefix in configs}
    if base_keys != set(expected):
        raise ValueError(
            f"Quantized artifact keys differ: missing={sorted(set(expected) - base_keys)}, extra={sorted(base_keys - set(expected))}"
        )
    for key, value in tensors.items():
        if key in owned:
            continue
        if (
            value.shape != expected[key].shape
            or not value.is_floating_point()
            or not torch.isfinite(value.float()).all()
        ):
            raise ValueError(f"Quantized artifact invalid floating tensor: {key}")
        cast = value.to(device=device, dtype=dtype if floating_dtypes is None else floating_dtypes.get(key, dtype))
        if not torch.isfinite(cast.float()).all():
            raise ValueError(f"Quantized artifact exceeds compute dtype: {key}")
        converted[key] = cast
    layers = []
    for name, compute, local in replacements:
        prefix = name + "."
        bias = converted.get(prefix + "bias")
        if bias is not None:
            local["bias"] = bias
        layer = QuantizedLinear(local, dtype=compute)
        layers.append((name, layer))
    for name, layer in layers:
        model.set_submodule(name, layer)
        converted.update((name + "." + key, value) for key, value in layer.state_dict().items())
    model.load_state_dict(converted, strict=True, assign=True)
    return True
