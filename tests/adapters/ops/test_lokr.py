"""Qualify dense Kronecker fusion without changing contraction or gradient reduction."""

import pytest
import torch

from unitunner.adapters.ops.lokr import scaled_kronecker


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("right_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_kronecker_rounding(right_dtype: torch.dtype, dtype: torch.dtype) -> None:
    torch.manual_seed(31)
    a = torch.randn(7, 5, device="cuda", requires_grad=True)
    b = torch.randn(11, 13, device="cuda", dtype=right_dtype, requires_grad=True)
    s = torch.tensor(0.1234567, device="cuda")
    actual = scaled_kronecker(a, b, s, dtype)
    expected = (torch.kron(a, b) * s).to(dtype)
    assert "Kronecker" in type(actual.grad_fn).__name__
    assert torch.equal(actual, expected)
    g = torch.randn_like(actual)
    ref = torch.autograd.grad(expected, (a, b), g)
    got = torch.autograd.grad(actual, (a, b), g)
    assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("broadcast", [False, True])
def test_kronecker_strided_gradient_preserves_reduction_layout(broadcast: bool) -> None:
    a = torch.randn(7, 5, device="cuda", requires_grad=True)
    b = torch.randn(11, 13, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scale = torch.tensor(0.1234567, device="cuda")
    actual = scaled_kronecker(a, b, scale, torch.bfloat16)
    expected = (torch.kron(a, b) * scale).bfloat16()
    gradient = (
        torch.randn(1, device="cuda", dtype=torch.bfloat16).expand_as(actual)
        if broadcast
        else torch.randn(65, 77, device="cuda", dtype=torch.bfloat16).T
    )
    ref = torch.autograd.grad(expected, (a, b), gradient)
    got = torch.autograd.grad(actual, (a, b), gradient)
    assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
def test_kronecker_higher_grad_and_fallback(device: str) -> None:
    a = torch.randn(3, 4, device=device, requires_grad=True)
    b = torch.randn(2, 5, device=device, requires_grad=True)
    s = torch.tensor(0.125, device=device)
    for left, scale in ((a, s), (a.T, s), (a, s.clone().requires_grad_())):
        actual = scaled_kronecker(left, b, scale, torch.float32)
        expected = torch.kron(left.contiguous(), b) * scale
        variables = (left, b, scale) if scale.requires_grad else (left, b)
        got = torch.autograd.grad(actual.sum(), variables, create_graph=True)
        ref = torch.autograd.grad(expected.sum(), variables, create_graph=True)
        assert torch.equal(actual, expected)
        assert all(torch.equal(x, y) for x, y in zip(got, ref, strict=True))
        assert torch.equal(torch.autograd.grad(got[0].sum(), b)[0], torch.autograd.grad(ref[0].sum(), b)[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kronecker_transforms_and_saved_versions() -> None:
    """Preserve transforms and reject changes to saved factors before backward."""
    a = torch.randn(3, 4, device="cuda", requires_grad=True)
    b = torch.randn(2, 5, device="cuda", requires_grad=True)
    s = torch.tensor(0.125, device="cuda")
    actual = torch.func.grad(lambda x: scaled_kronecker(x, b, s, torch.float32).sum())(a)
    expected = torch.func.grad(lambda x: (torch.kron(x, b) * s).sum())(a)
    assert torch.equal(actual, expected)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(a, torch.ones_like(a))
        actual = torch.autograd.forward_ad.unpack_dual(scaled_kronecker(dual, b, s, torch.float32)).tangent
        expected = torch.autograd.forward_ad.unpack_dual(torch.kron(dual, b) * s).tangent
        assert actual is not None and expected is not None
        assert torch.equal(actual, expected)
    compiled = torch.compile(scaled_kronecker, backend="eager", fullgraph=True)
    assert torch.equal(compiled(a, b, s, torch.float32), torch.kron(a, b) * s)
    output = scaled_kronecker(a, b, s, torch.float32)
    with torch.no_grad():
        b.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()
