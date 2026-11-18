"""Verify joint frozen-base LoRA, live scales and every native derivative."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck
from torch.nn import functional as F

from unituner.adapters.ops.lora import _scaled_residual, fused_lora_linear


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fused_input_and_factor_gradients_three_updates(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(293)
    weight = torch.randn(11, 7, device=device)
    bias = torch.randn(11, device=device)
    values = [torch.randn(*shape, device=device) for shape in ((2, 3, 7), (3, 7), (11, 3))]
    groups = [[torch.nn.Parameter(value.clone()) for value in values] for _ in range(2)]
    optimizers = [torch.optim.AdamW(group, lr=0.001) for group in groups]
    for scale_value in (1.5, 0.75, 2.0):
        scale = weight.new_tensor(scale_value)
        x, down, up = groups[0]
        actual = fused_lora_linear(x, weight, bias, down, up, scale)
        x, down, up = groups[1]
        expected = F.linear(x, weight, bias) + F.linear(F.linear(x, down), up) * scale
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for prediction, optimizer in zip((actual, expected), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            prediction.square().mean().backward()
        for a, b in zip(*groups, strict=True):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4)
        for optimizer in optimizers:
            optimizer.step()
        for a, b in zip(*groups, strict=True):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


def test_fused_higher_derivatives():
    torch.manual_seed(294)
    weight = torch.randn(4, 3, dtype=torch.float64)
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    down = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    up = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    scale = weight.new_tensor(1.5)

    def function(x, down, up):
        return fused_lora_linear(x, weight, None, down, up, scale)

    assert gradcheck(function, (x, down, up))
    assert gradgradcheck(function, (x, down, up))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_residual_preserves_scaling_rounding_and_strided_gradient(dtype):
    base = torch.randn(129, 65, device="cuda", dtype=dtype, requires_grad=True)
    delta = torch.randn_like(base, requires_grad=True)
    scale = torch.tensor(-0.1234567, device="cuda")
    actual = _scaled_residual(base, delta, scale)
    expected = base + delta * scale
    assert "ScaledResidual" in type(actual.grad_fn).__name__
    assert torch.equal(actual, expected)
    gradient = torch.randn(65, 129, device="cuda", dtype=dtype).T
    got = torch.autograd.grad(actual, (base, delta), gradient)
    ref = torch.autograd.grad(expected, (base, delta), gradient)
    assert all(torch.equal(a, b) for a, b in zip(got, ref, strict=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_residual_higher_derivatives_transforms_and_live_scale():
    base = torch.randn(64, 64, device="cuda", requires_grad=True)
    delta = torch.randn_like(base, requires_grad=True)
    scale = torch.tensor(0.1234567, device="cuda")
    actual = _scaled_residual(base, delta, scale)
    expected = base + delta * scale
    got = torch.autograd.grad(actual.square().sum(), delta, create_graph=True)[0]
    ref = torch.autograd.grad(expected.square().sum(), delta, create_graph=True)[0]
    assert torch.equal(got, ref)
    assert torch.equal(torch.autograd.grad(got.sum(), base)[0], torch.autograd.grad(ref.sum(), base)[0])
    transformed = torch.func.grad(lambda value: _scaled_residual(value, delta, scale).sum())(base)
    assert torch.equal(transformed, torch.ones_like(base))
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(base, torch.ones_like(base))
        tangent = torch.autograd.forward_ad.unpack_dual(_scaled_residual(dual, delta, scale)).tangent
        assert tangent is not None and torch.equal(tangent, torch.ones_like(base))
    compiled = torch.compile(_scaled_residual, backend="eager", fullgraph=True)
    assert torch.equal(compiled(base, delta, scale), expected)
    scale.requires_grad_()
    got = torch.autograd.grad(_scaled_residual(base, delta, scale).sum(), scale)[0]
    ref = torch.autograd.grad((base + delta * scale).sum(), scale)[0]
    assert torch.equal(got, ref)
    scale.requires_grad_(False)
    actual = _scaled_residual(base, delta, scale)
    with torch.no_grad():
        scale.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        actual.sum().backward()
