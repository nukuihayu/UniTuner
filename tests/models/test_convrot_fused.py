"""Validate optional fused ConvRot compute against its packed-weight baseline."""

import pytest
import torch
from torch import nn

from unituner.artifacts.quantization import quantize_weight, serialize_weight
from unituner.config.execution import ExecutionConfig
from unituner.execution.installation import prepare_base_weights, validate_training_execution
from unituner.models.components import quantized
from unituner.models.components.quantized import QuantizedLinear


def packed_layer(n: int, k: int, group: int, device: str = "cpu") -> QuantizedLinear:
    weight = quantize_weight(torch.randn(n, k, device=device) * 0.03, "int8_convrot", rotation_size=group)
    state = serialize_weight("weight", weight, "int8_convrot")
    state["bias"] = torch.randn(n, device=device) * 0.03
    return QuantizedLinear(state, dtype=torch.bfloat16)


@pytest.mark.parametrize("output_limit", (24, 1 << 26))
def test_large_convrot_row_chunks_preserve_forward_and_backward(monkeypatch, output_limit):
    from comfy_kitchen.backends.triton import quantization
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

    def fake_int8_linear(inputs, weight, scale, bias, dtype, rotate, group_size):
        values = inputs.reshape(-1, inputs.shape[-1])
        if rotate:
            values = _rotate_activation(values, _build_hadamard(group_size, dtype=values.dtype), group_size)
        output = values @ (weight.to(values.dtype) * scale.reshape(-1, 1).to(values.dtype)).T
        if bias is not None:
            output = output + bias
        return output.reshape(*inputs.shape[:-1], weight.shape[0])

    monkeypatch.setattr(quantization, "int8_linear", fake_int8_linear)
    weight = torch.randint(-10, 10, (8, 8), dtype=torch.int8)
    scale = torch.ones(8)
    values = torch.randn(2, 7, 8, requires_grad=True)
    baseline = quantized._FusedConvRotLinear.apply(values, weight, scale, None, 4, "int8")
    baseline.square().sum().backward()
    baseline_grad = values.grad.clone()
    values.grad = None
    # The address limit applies to row offsets, not to the frozen weight itself.
    monkeypatch.setattr(quantized, "_CONVROT_ADDRESS_LIMIT", 64)
    monkeypatch.setattr(quantized, "_CONVROT_OUTPUT_LIMIT", output_limit)
    monkeypatch.setattr(quantized, "_CONVROT_SINGLE_CALL_LIMIT", 24)
    chunked = quantized._FusedConvRotLinear.apply(values, weight, scale, None, 4, "int8")
    chunked.square().sum().backward()
    torch.testing.assert_close(chunked, baseline)
    torch.testing.assert_close(values.grad, baseline_grad)


def test_convrot_safe_single_call_avoids_extra_output_buffer(monkeypatch):
    from comfy_kitchen.backends.triton import quantization

    calls = []

    def fake_int8_linear(inputs, weight, scale, bias, dtype, rotate, group_size):
        calls.append(inputs.shape)
        return torch.zeros((*inputs.shape[:-1], weight.shape[0]), dtype=dtype)

    monkeypatch.setattr(quantization, "int8_linear", fake_int8_linear)
    monkeypatch.setattr(quantized, "_CONVROT_OUTPUT_LIMIT", 24)
    monkeypatch.setattr(quantized, "_CONVROT_SINGLE_CALL_LIMIT", 128)
    inputs = torch.zeros(2, 7, 8)
    result = quantized._convrot_int8_linear(
        inputs, torch.zeros(8, 8, dtype=torch.int8), torch.ones(8), None, 4, rotate=True
    )
    assert result.shape == (2, 7, 8)
    assert calls == [inputs.shape]


def test_convrot_auto_falls_back_on_cpu_and_forced_kernel_rejects_cpu():
    layer = packed_layer(32, 64, 4)
    layer.configure_convrot("auto")
    inputs = torch.randn(3, 64, requires_grad=True)
    expected = layer(inputs)
    expected.sum().backward()
    assert torch.isfinite(inputs.grad).all()
    layer.configure_convrot("triton")
    with pytest.raises(RuntimeError, match="CUDA and Triton"):
        layer(inputs)


def test_convrot_execution_policy_requires_packed_base_and_cuda():
    with pytest.raises(ValueError, match="fused ConvRot kernel"):
        ExecutionConfig(convrot_backward="int8")
    config = ExecutionConfig(convrot_kernel="triton")
    with pytest.raises(ValueError, match="CUDA"):
        validate_training_execution(config, device=torch.device("cpu"), adapter=True)
    with pytest.raises(ValueError, match="ConvRot INT8 base weights"):
        prepare_base_weights(nn.Linear(64, 32), config)
    model = nn.Sequential(packed_layer(32, 64, 4))
    prepare_base_weights(model, config)
    assert model[0]._convrot_kernel == "triton"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ConvRot")
@pytest.mark.parametrize(("m", "n", "k", "group"), [(1, 32, 64, 4), (17, 64, 256, 64), (33, 128, 512, 256)])
@pytest.mark.parametrize("backward", ("float", "int8"))
def test_fused_convrot_shapes_and_adapter_gradients(m, n, k, group, backward):
    pytest.importorskip("triton")
    torch.manual_seed(41)
    layer = packed_layer(n, k, group, "cuda")
    input_ref = torch.randn(2, m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    input_fused = input_ref.detach().clone().requires_grad_()
    adapter_ref = nn.Linear(k, 8, bias=False, device="cuda", dtype=torch.bfloat16)
    adapter_fused = nn.Linear(k, 8, bias=False, device="cuda", dtype=torch.bfloat16)
    adapter_fused.load_state_dict(adapter_ref.state_dict())
    up_ref = nn.Linear(8, n, bias=False, device="cuda", dtype=torch.bfloat16)
    up_fused = nn.Linear(8, n, bias=False, device="cuda", dtype=torch.bfloat16)
    up_fused.load_state_dict(up_ref.state_dict())

    reference = layer(input_ref) + up_ref(adapter_ref(input_ref))
    reference.float().square().mean().backward()
    layer.configure_convrot("triton", backward)
    fused = layer(input_fused) + up_fused(adapter_fused(input_fused))
    fused.float().square().mean().backward()
    # Dynamic row-wise activation quantization adds error beyond packed-weight error.
    assert (fused.float() - reference.float()).norm() / reference.float().norm() < 0.06
    assert (input_fused.grad.float() - input_ref.grad.float()).norm() / input_ref.grad.float().norm() < 0.09
    torch.testing.assert_close(adapter_fused.weight.grad, adapter_ref.weight.grad, atol=1e-3, rtol=0.09)
    torch.testing.assert_close(up_fused.weight.grad, up_ref.weight.grad, atol=1e-3, rtol=0.09)
    assert layer.weight.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ConvRot")
@pytest.mark.parametrize("backward", ("float", "int8"))
def test_fused_convrot_cuda_row_chunks(monkeypatch, backward):
    pytest.importorskip("triton")
    layer = packed_layer(32, 64, 4, "cuda")
    layer.configure_convrot("triton", backward)
    source = torch.randn(2, 17, 64, dtype=torch.bfloat16, device="cuda")
    regular = source.detach().clone().requires_grad_()
    baseline = layer(regular)
    baseline.float().square().sum().backward()
    monkeypatch.setattr(quantized, "_CONVROT_OUTPUT_LIMIT", 128)
    monkeypatch.setattr(quantized, "_CONVROT_SINGLE_CALL_LIMIT", 128)
    chunked_input = source.detach().clone().requires_grad_()
    chunked = layer(chunked_input)
    chunked.float().square().sum().backward()
    torch.testing.assert_close(chunked, baseline)
    torch.testing.assert_close(chunked_input.grad, regular.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ConvRot")
def test_fused_convrot_autocast_and_replay():
    pytest.importorskip("triton")
    layer = packed_layer(64, 256, 64, "cuda")
    layer.configure_convrot("triton")
    inputs = torch.randn(2, 256, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = torch.utils.checkpoint.checkpoint(layer, inputs, use_reentrant=False)
    assert result.dtype == torch.bfloat16
    result.float().square().mean().backward()
    assert inputs.grad.dtype == inputs.dtype and torch.isfinite(inputs.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton ConvRot")
def test_fused_convrot_fp32_input():
    pytest.importorskip("triton")
    layer = packed_layer(64, 256, 64, "cuda")
    layer.bias = None
    inputs = torch.randn(4, 256, device="cuda", requires_grad=True)
    expected = layer(inputs)
    layer.configure_convrot("triton")
    actual = layer(inputs)
    assert actual.dtype == inputs.dtype
    assert (actual - expected).norm() / expected.norm() < 0.04
    actual.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
