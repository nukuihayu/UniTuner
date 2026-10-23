"""Qualify fused products against native dtype promotion and autograd."""

import pytest
import torch

from unitunner.adapters.ops.loha import scaled_product


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_product_rounding(dtype: torch.dtype) -> None:
    """Non-unit FP32 scalars must respect low-precision intermediate casts."""
    torch.manual_seed(120)
    a = torch.randn(37, 71, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    s = torch.tensor(-0.1234567, device="cuda")
    g = torch.randn(71, 37, device="cuda", dtype=dtype).T
    expected = (a * b) * s
    actual = scaled_product(a, b, s)
    assert "ScaledProduct" in type(actual.grad_fn).__name__
    assert torch.equal(actual, expected)
    reference = torch.autograd.grad(expected, (a, b), g)
    result = torch.autograd.grad(actual, (a, b), g)
    assert all(torch.equal(x, y) for x, y in zip(result, reference, strict=True))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
def test_product_fallback_and_higher_derivatives(device: str) -> None:
    """Trainable scalars, broadcasting, views and derivative transforms stay native."""
    a = torch.randn(3, 5, device=device, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    s = torch.tensor(0.25, device=device)
    for left, right, scale in ((a, b, s), (a.T, b.T, s), (a, b[:1], s), (a, b, s.clone().requires_grad_())):
        expected = (left * right) * scale
        actual = scaled_product(left, right, scale)
        variables = (left, right, scale) if scale.requires_grad else (left, right)
        ref = torch.autograd.grad(expected.sum(), variables, create_graph=True)
        got = torch.autograd.grad(actual.sum(), variables, create_graph=True)
        assert torch.equal(actual, expected)
        assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True))
        assert torch.equal(torch.autograd.grad(got[0].sum(), right)[0], torch.autograd.grad(ref[0].sum(), right)[0])
    assert torch.equal(torch.func.grad(lambda x: scaled_product(x, b, s).sum())(a), b * s)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(a, torch.ones_like(a))
        tangent = torch.autograd.forward_ad.unpack_dual(scaled_product(dual, b, s)).tangent
        assert tangent is not None
        assert torch.equal(tangent, b * s)
    compiled = torch.compile(scaled_product, backend="eager", fullgraph=True)
    assert torch.equal(compiled(a, b, s), (a * b) * s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_product_saved_values_and_zero_stride_gradient() -> None:
    """Save versioned factors and accept expanded gradients from scalar losses."""
    a = torch.randn(7, 11, device="cuda", requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    s = torch.tensor(0.7, device="cuda")
    actual = scaled_product(a, b, s)
    got = torch.autograd.grad(actual.sum(), (a, b))
    ref = torch.autograd.grad(((a * b) * s).sum(), (a, b))
    assert all(torch.equal(x, y) for x, y in zip(got, ref, strict=True))
    changed = scaled_product(a, b, s)
    with torch.no_grad():
        s.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        changed.sum().backward()
