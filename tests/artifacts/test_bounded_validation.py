"""Check bounded finite validation across storage layouts and dtypes."""

import pytest
import torch

from unituner.artifacts.validation import all_finite


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float8_e4m3fn, torch.int8])
@pytest.mark.parametrize("transposed", [False, True])
def test_chunked_finite_check(dtype, transposed):
    value = torch.ones(16, 32).to(dtype)
    if transposed:
        value = value.T
    assert all_finite(value, chunk_elements=40)
    if dtype != torch.int8:
        value[1, 2] = float("nan")
        assert not all_finite(value, chunk_elements=40)


def test_finite_validation_never_expands_the_whole_projection(monkeypatch):
    original = torch.isfinite
    sizes = []

    def check(value):
        sizes.append(value.numel())
        return original(value)

    monkeypatch.setattr(torch, "isfinite", check)
    assert all_finite(torch.ones(256, 256, dtype=torch.bfloat16), chunk_elements=1024)
    assert len(sizes) == 64 and max(sizes) == 1024
    assert all_finite(torch.empty(0))
    with pytest.raises(ValueError, match="positive"):
        all_finite(torch.ones(1), chunk_elements=0)


def test_noncontiguous_validation_and_casts_stay_bounded(monkeypatch):
    original = torch.isfinite
    sizes = []

    def check(value):
        sizes.append(value.numel())
        return original(value)

    monkeypatch.setattr(torch, "isfinite", check)
    values = torch.ones(1, 64, 64).transpose(0, 1)
    assert all_finite(values, dtype=torch.float16, chunk_elements=16)
    assert max(sizes) <= 16
    values[0, 0, 0] = 1e10
    assert not all_finite(values, dtype=torch.float16, chunk_elements=16)
