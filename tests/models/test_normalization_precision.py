"""Guard model-specific FP32 cast sharing against BF16 backward regressions."""

import pytest
import torch

from unituner.models.moss_image.layers import RMSNorm as MossNorm
from unituner.models.wan.layers import RMSNorm as WanNorm

DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("weight_dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("family", ("wan", "moss"))
def test_model_specific_cast_branches_match_reference_backward(device, weight_dtype, family):
    torch.manual_seed(953)
    norm = (WanNorm(16, 1e-5) if family == "wan" else MossNorm(16)).to(device=device, dtype=weight_dtype)
    values = torch.randn(2, 11, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    other = values.detach().clone().requires_grad_()
    weight = norm.weight.detach().clone().requires_grad_()
    # Musubi Wan casts once before _norm; published MOSS _RMSNorm casts
    # each branch independently. These have different BF16 VJPs.
    if family == "wan":
        floating = other.float()
        expected = (floating * torch.rsqrt(floating.pow(2).mean(-1, keepdim=True) + 1e-5)).to(other.dtype)
        expected = expected * weight.to(other.dtype)
    else:
        expected = other.float() * torch.rsqrt(other.float().square().mean(-1, keepdim=True) + 1e-6)
        expected = expected.to(other.dtype) * weight
    actual = norm(values)
    upstream = torch.randn_like(expected)
    observed = torch.autograd.grad(actual, (values, norm.weight), upstream)
    reference = torch.autograd.grad(expected, (other, weight), upstream)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(observed, reference, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
