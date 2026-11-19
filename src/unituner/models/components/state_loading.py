"""Validate native model state before assigning floating or packed storage."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from unituner.artifacts.quantized_loading import load_quantized_model
from unituner.artifacts.validation import all_finite


def load_model_state(
    model: nn.Module,
    tensors: Mapping[str, Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype | None,
    label: str,
    floating_dtypes: Mapping[str, torch.dtype] | None = None,
    check_finite: bool = True,
    quantization_family: str | None = None,
) -> None:
    """Validate the full inventory before mutation, preserving per-parameter storage policy.

    Requested-dtype range checks precede assignment, including for protected FP32
    islands, matching native loader behavior. Z-Image's historical floating path
    checks geometry only; packed artifacts always retain their strict validation.
    """
    if quantization_family is not None:
        from unituner.artifacts.formats.precision_policy import language_boundaries, protection_reason

        keys = tuple(model.state_dict())
        boundaries = language_boundaries(keys)
        pruned_adaln = "adaln_t_table" in keys or "adaln_basis" in keys
        floating_dtypes = dict(floating_dtypes or {})
        for name, layer in model.named_modules():
            key = name + ".weight"
            value = tensors.get(key)
            if (
                isinstance(layer, nn.Linear)
                and value is not None
                and value.dtype in (torch.float32, torch.float16, torch.bfloat16)
                and name + ".comfy_quant" not in tensors
                and not protection_reason(quantization_family, key, boundaries, pruned_adaln=pruned_adaln)
            ):
                # The quantizer casts one layer to FP32. Retain its mmap/source
                # storage until then rather than expanding the entire model.
                floating_dtypes[key] = value.dtype
    if load_quantized_model(model, tensors, device=device, dtype=dtype, floating_dtypes=floating_dtypes):
        return
    expected = model.state_dict()
    if expected.keys() != tensors.keys():
        raise ValueError(
            f"{label} artifact keys differ: missing={sorted(expected.keys() - tensors.keys())}, "
            f"extra={sorted(tensors.keys() - expected.keys())}"
        )
    for name, value in tensors.items():
        if value.shape != expected[name].shape or not value.is_floating_point():
            raise ValueError(f"{label} invalid tensor shape/type; values invalid: {name}")
        if check_finite:
            if not all_finite(value):
                raise ValueError(f"{label} invalid tensor; values invalid: {name}")
            # An unchanged dtype cannot introduce conversion overflow.
            if dtype is not None and dtype != value.dtype and not all_finite(value, dtype=dtype):
                raise ValueError(f"{label} tensor exceeds requested dtype range: {name}")
            storage_dtype = dtype if floating_dtypes is None else floating_dtypes.get(name, dtype)
            if storage_dtype is not None and storage_dtype != dtype and not all_finite(value, dtype=storage_dtype):
                raise ValueError(f"{label} tensor exceeds storage dtype range: {name}")
    model.load_state_dict(
        {
            name: value.to(device=device, dtype=dtype if floating_dtypes is None else floating_dtypes.get(name, dtype))
            for name, value in tensors.items()
        },
        strict=True,
        assign=True,
    )
