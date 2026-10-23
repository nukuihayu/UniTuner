"""Prepare frozen base projections before adapter and optimizer installation."""

import torch
from torch import nn

from unitunner.artifacts.formats.precision_policy import language_boundaries, protection_reason
from unitunner.artifacts.formats.quantization import DEFAULT_GROUP_SIZE, DEFAULT_ROTATION_SIZE, FORMATS
from unitunner.artifacts.quantization import quantize_weight, serialize_weight
from unitunner.models.components.quantized import QuantizedLinear


def quantize_model(
    model: nn.Module,
    precision: str,
    *,
    family: str,
    rotation_size: int = DEFAULT_ROTATION_SIZE,
    group_size: int = DEFAULT_GROUP_SIZE,
    device: torch.device | None = None,
) -> int:
    """Quantize one projection at a time without modifying the source checkpoint.

    Incompatible shapes fail explicitly. Existing quantized layers must match the
    requested format; re-quantizing a lossy base implicitly is never permitted.
    """
    if precision not in FORMATS:
        raise ValueError("Training quantization requires a known Kitchen format")
    boundaries = language_boundaries(model.state_dict())
    count = 0
    for name, layer in tuple(model.named_modules()):
        if not isinstance(layer, nn.Linear) or protection_reason(family, name + ".weight", boundaries):
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
                layer.weight.to(device=quantization_device),
                precision,
                rotation_size=rotation_size,
                group_size=group_size,
            )
            state = serialize_weight("weight", weight, precision)
            if layer.bias is not None:
                state["bias"] = layer.bias.detach().to(weight.device)
            replacement = QuantizedLinear(state, dtype=layer.weight.dtype).to(device=layer.weight.device)
        model.set_submodule(name, replacement)
        count += 1
    if not count:
        raise ValueError(f"{family}: no eligible projections for {precision}")
    return count
