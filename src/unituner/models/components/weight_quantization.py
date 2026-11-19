"""Prepare frozen base projections before adapter and optimizer installation."""

import torch
from torch import nn

from unituner.artifacts.formats.precision_policy import language_boundaries, protection_reason
from unituner.artifacts.formats.quantization import DEFAULT_GROUP_SIZE, DEFAULT_ROTATION_SIZE, FORMATS
from unituner.artifacts.quantization import quantize_weight, serialize_weight
from unituner.models.components.quantized import QuantizedLinear


def quantize_model(
    model: nn.Module,
    precision: str,
    *,
    family: str,
    rotation_size: int = DEFAULT_ROTATION_SIZE,
    group_size: int = DEFAULT_GROUP_SIZE,
    device: torch.device | None = None,
    input_dtype: torch.dtype | None = None,
) -> int:
    """Quantize one projection at a time without modifying the source checkpoint.

    Incompatible shapes fail explicitly. Existing quantized layers must match the
    requested format; re-quantizing a lossy base implicitly is never permitted.
    """
    if precision not in FORMATS:
        raise ValueError("Training quantization requires a known Kitchen format")
    keys = tuple(model.state_dict())
    boundaries = language_boundaries(keys)
    pruned_adaln = "adaln_t_table" in keys or "adaln_basis" in keys
    count = 0
    # Keep names, not a snapshot of modules/state tensors: replaced dense
    # projections must be collectable before quantizing the next projection.
    names = tuple(name for name, layer in model.named_modules() if isinstance(layer, nn.Linear))
    for name in names:
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear) or protection_reason(
            family, name + ".weight", boundaries, pruned_adaln=pruned_adaln
        ):
            continue
        if isinstance(layer, QuantizedLinear):
            try:
                layer.validate_quantization(precision, rotation_size=rotation_size, group_size=group_size)
            except ValueError as error:
                raise ValueError(f"{name}: {error}") from error
            count += 1
            continue
        if FORMATS[precision].calibrated:
            raise ValueError(
                f"{name}: {precision} requires an already calibrated quantized artifact; "
                "automatic training-time quantization cannot invent calibration statistics"
            )
        if layer.weight.is_meta or layer.weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"{name}: quantization requires materialized FP32/FP16/BF16 weights")
        quantization_device = device or layer.weight.device
        with torch.no_grad(), torch.autocast(quantization_device.type, enabled=False):
            weight = quantize_weight(
                layer.weight.to(device=quantization_device, dtype=input_dtype),
                precision,
                rotation_size=rotation_size,
                group_size=group_size,
            )
            state = serialize_weight("weight", weight, precision)
            if layer.bias is not None:
                state["bias"] = layer.bias.detach().to(weight.device)
            replacement = QuantizedLinear(state, dtype=input_dtype or layer.weight.dtype).to(device=layer.weight.device)
        model.set_submodule(name, replacement)
        count += 1
        del layer, weight, state, replacement
    if not count:
        raise ValueError(f"{family}: no eligible projections for {precision}")
    return count
