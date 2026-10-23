"""Qualify fused rotary values, derivatives, and native fallback contracts."""

import pytest
import torch
from torch import Tensor

from unitunner.models.components.rotary import apply_rotary


def reference(values: Tensor, positions: Tensor) -> Tensor:
    """Evaluate the original independent products, FP32 sum, and final cast."""
    pairs = values.float().reshape(*values.shape[:-1], -1, 1, 2)
    result = positions[..., 0] * pairs[..., 0] + positions[..., 1] * pairs[..., 1]
    return result.reshape_as(values).to(values)


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    return torch.device("cuda")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["blhd", "bhld", "sliced", "expanded"])
def test_cuda_values_and_strided_backward_are_bitwise_exact(cuda, dtype, layout):
    torch.manual_seed(913)
    if layout == "blhd":
        inputs = torch.randn(2, 37, 3, 64, device=cuda, dtype=dtype).transpose(1, 2)
    elif layout == "sliced":
        inputs = torch.randn(2, 3, 37, 128, device=cuda, dtype=dtype)[..., ::2]
    elif layout == "expanded":
        inputs = torch.randn(1, 3, 37, 64, device=cuda, dtype=dtype).expand(2, -1, -1, -1)
    else:
        inputs = torch.randn(2, 3, 37, 64, device=cuda, dtype=dtype)
    inputs.requires_grad_()
    other = inputs.detach().clone().requires_grad_()
    positions = torch.randn(2, 1, 37, 32, 2, 2, device=cuda)
    actual, expected = apply_rotary(inputs, positions), reference(other, positions)
    gradient = torch.randn(2, 3, 37, 64, 2, device=cuda, dtype=dtype)[..., 0]
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(inputs.grad, other.grad, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["trainable", "transposed", "broadcast", "float64"])
def test_native_position_contracts(device, kind):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(914)
    inputs = torch.randn(2, 3, 7, 16, device=device, requires_grad=True)
    positions = torch.randn(2, 1, 7, 8, 2, 2, device=device)
    if kind == "transposed":
        positions = positions.transpose(-1, -2)
    elif kind == "broadcast":
        positions = positions[:1]
    elif kind == "float64":
        positions = positions.double()
    if kind == "trainable":
        positions.requires_grad_()
    variables = (inputs, positions) if kind == "trainable" else (inputs,)
    actual, expected = apply_rotary(inputs, positions), reference(inputs, positions)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_grad = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), variables)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_cuda_higher_derivatives_and_zero_stride_gradient(cuda):
    torch.manual_seed(915)
    inputs = torch.randn(1, 2, 7, 16, device=cuda, requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    positions = torch.randn(1, 1, 7, 8, 2, 2, device=cuda)
    first = torch.autograd.grad(apply_rotary(inputs, positions).square().sum(), inputs, create_graph=True)[0]
    expected = torch.autograd.grad(reference(other, positions).square().sum(), other, create_graph=True)[0]
    torch.testing.assert_close(first, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(first.sum(), inputs)[0], torch.autograd.grad(expected.sum(), other)[0], atol=0, rtol=0
    )
    torch.testing.assert_close(
        torch.autograd.grad(apply_rotary(inputs, positions).sum(), inputs)[0],
        torch.autograd.grad(reference(other, positions).sum(), other)[0],
        atol=0,
        rtol=0,
    )


def test_cuda_saved_matrix_version_check(cuda):
    inputs = torch.randn(1, 2, 7, 16, device=cuda, requires_grad=True)
    positions = torch.randn(1, 1, 7, 8, 2, 2, device=cuda)
    result = apply_rotary(inputs, positions)
    positions.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()


def test_cuda_compiler_and_function_transforms_keep_native_path(cuda):
    inputs = torch.randn(1, 2, 7, 16, device=cuda, requires_grad=True)
    positions = torch.randn(1, 1, 7, 8, 2, 2, device=cuda)
    compiled = torch.compile(apply_rotary, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(inputs, positions), reference(inputs, positions), atol=0, rtol=0)
    actual = torch.func.grad(lambda x: apply_rotary(x, positions).square().sum())(inputs)
    expected = torch.func.grad(lambda x: reference(x, positions).square().sum())(inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    tangent = torch.randn_like(positions)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(positions, tangent)
        actual = torch.autograd.forward_ad.unpack_dual(apply_rotary(inputs, dual))
        expected = torch.autograd.forward_ad.unpack_dual(reference(inputs, dual))
        torch.testing.assert_close(actual.primal, expected.primal, atol=0, rtol=0)
        torch.testing.assert_close(actual.tangent, expected.tangent, atol=0, rtol=0)
