"""Validate native model state before assigning floating or packed storage."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from unitunner.artifacts.quantized_loading import load_quantized_model


def load_model_state(
    model: nn.Module,
    tensors: Mapping[str, Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype | None,
    label: str,
    floating_dtypes: Mapping[str, torch.dtype] | None = None,
    check_finite: bool = True,
) -> None:
    """Validate the full inventory before mutation, preserving per-parameter storage policy.

    Requested-dtype range checks precede assignment, including for protected FP32
    islands, matching native loader behavior. Z-Image's historical floating path
    checks geometry only; packed artifacts always retain their strict validation.
    """
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
            if not torch.isfinite(value).all():
                raise ValueError(f"{label} invalid tensor; values invalid: {name}")
            # An unchanged dtype cannot introduce conversion overflow.
            if dtype is not None and dtype != value.dtype and not torch.isfinite(value.to(dtype=dtype)).all():
                raise ValueError(f"{label} tensor exceeds requested dtype range: {name}")
    model.load_state_dict(
        {
            name: value.to(device=device, dtype=dtype if floating_dtypes is None else floating_dtypes.get(name, dtype))
            for name, value in tensors.items()
        },
        strict=True,
        assign=True,
    )
