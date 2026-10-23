"""Verify shared loading validates before replacing model storage."""

import pytest
import torch
from torch import nn

from unitunner.models.components.state_loading import load_model_state
from unitunner.precision import compute_context


@pytest.mark.parametrize("invalid", ["missing", "shape", "integer", "nan", "overflow"])
def test_invalid_state_does_not_mutate_parameters(invalid):
    model = nn.Linear(2, 2)
    before = tuple(model.parameters())
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    if invalid == "missing":
        del state["bias"]
    elif invalid == "shape":
        state["bias"] = torch.ones(3)
    elif invalid == "integer":
        state["bias"] = torch.ones(2, dtype=torch.int64)
    else:
        state["bias"].fill_(float("nan") if invalid == "nan" else 1e10)
    with pytest.raises(ValueError):
        load_model_state(model, state, device=torch.device("cpu"), dtype=torch.float16, label="test")
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))


def test_per_parameter_storage_policy():
    model = nn.Linear(2, 2)
    state = model.state_dict()
    load_model_state(
        model,
        state,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        label="test",
        floating_dtypes={"bias": torch.float32},
    )
    assert model.weight.dtype == torch.bfloat16
    assert model.bias.dtype == torch.float32


@pytest.mark.parametrize("dtype", [None, torch.float32])
def test_unchanged_dtype_rejects_nonfinite_before_assignment(dtype):
    model = nn.Linear(2, 2)
    before = tuple(model.parameters())
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    state["bias"].fill_(float("nan"))
    with pytest.raises(ValueError, match="invalid tensor"):
        load_model_state(model, state, device=torch.device("cpu"), dtype=dtype, label="test")
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))


@pytest.mark.parametrize(
    "precision, dtype", [("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)]
)
def test_compute_context_overrides_and_restores_outer_precision(precision, dtype):
    value = torch.randn(3, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with compute_context(value.device, precision):
            assert (value @ value).dtype == dtype
        assert (value @ value).dtype == torch.bfloat16
