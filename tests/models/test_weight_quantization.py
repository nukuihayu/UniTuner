"""Check packed training against the exact dequantized-weight mathematical model."""

from contextlib import closing

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from unitunner.adapters.config import AdapterConfig
from unitunner.artifacts.formats.quantization import FORMATS
from unitunner.artifacts.quantization import quantize_weight
from unitunner.artifacts.quantized_loading import load_quantized_model
from unitunner.config.text import TextEncodingConfig
from unitunner.models.components.quantized import QuantizedLinear
from unitunner.models.components.text_loading import load_text_state, prepare_text_model
from unitunner.models.components.weight_quantization import quantize_model
from unitunner.training.parameters import ParameterInstallation

FORMATS_TO_TEST = tuple(name for name, spec in FORMATS.items() if not spec.calibrated)
DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


@pytest.mark.parametrize("precision", FORMATS_TO_TEST)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_storage_training_forward_backward_and_reload(precision, device, dtype):
    kitchen = pytest.importorskip("comfy_kitchen")
    torch.manual_seed(75)
    # Non-square widths catch accidental packed-storage geometry in adapters.
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(256, 128, device=device, dtype=dtype)])
    original = model.get_parameter("layers.0.weight").detach().clone()
    with kitchen.use_backend("eager"):
        expected = quantize_weight(original, precision).dequantize().to(dtype)
        assert quantize_model(model, precision, family="z-image") == 1
        layer = model.layers[0]
        assert isinstance(layer, QuantizedLinear)
        inputs = torch.randn(2, 3, 256, device=device, dtype=dtype, requires_grad=True)
        reference = inputs.detach().clone().requires_grad_()
        gradient = torch.randn(2, 3, 128, device=device, dtype=dtype)
        actual = layer(inputs)
        target = F.linear(reference, expected, layer.bias)
        torch.testing.assert_close(actual, target, atol=0, rtol=0)
        actual.backward(gradient)
        target.backward(gradient)
        torch.testing.assert_close(inputs.grad, reference.grad, atol=0, rtol=0)
        assert all(p.grad is None for p in layer.parameters())
        storage = {name: p.detach().clone() for name, p in layer.named_parameters() if name != "bias"}
        layer.to(dtype=torch.bfloat16)
        for name, value in storage.items():
            actual_value = getattr(layer, name)
            assert actual_value.dtype == value.dtype
            assert torch.equal(actual_value.reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8))
        layer.to(dtype=dtype)
        with torch.device("meta"):
            restored = nn.Module()
            restored.layers = nn.ModuleList([nn.Linear(256, 128)])
        assert load_quantized_model(restored, model.state_dict(), device=torch.device(device), dtype=dtype)
        torch.testing.assert_close(restored.layers[0](inputs), layer(inputs), atol=0, rtol=0)
        for algorithm in ("lora", "loha", "lokr"):
            with closing(
                ParameterInstallation(
                    model, adapter=AdapterConfig(algorithm=algorithm, rank=2, alpha=2.0), targets={"layers.0": layer}
                )
            ) as installation:
                optimizer = torch.optim.SGD(installation.plan.parameters, lr=0.01)
                before = tuple(p.detach().clone() for p in installation.plan.parameters)
                layer(inputs.detach()).float().square().mean().backward()
                optimizer.step()
                assert any(not torch.equal(a, b) for a, b in zip(before, installation.plan.parameters, strict=True))


def test_text_policy_protects_boundaries_towers_and_embeddings():
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)])
    model.visual = nn.Linear(16, 16)
    model.embed_tokens = nn.Embedding(16, 16)
    with pytest.importorskip("comfy_kitchen").use_backend("eager"):
        assert quantize_model(model, "float8_e4m3fn", family="qwen3-vl") == 2
    assert not isinstance(model.layers[0], QuantizedLinear)
    assert not isinstance(model.layers[-1], QuantizedLinear)
    assert isinstance(model.layers[1], QuantizedLinear)
    assert not isinstance(model.visual, QuantizedLinear)


@pytest.mark.parametrize("precision", FORMATS_TO_TEST)
def test_existing_base_reuse_preserves_storage_without_reparsing_metadata(precision, monkeypatch):
    kitchen = pytest.importorskip("comfy_kitchen")
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(256, 128)])
    with kitchen.use_backend("eager"):
        assert quantize_model(model, precision, family="z-image") == 1
    layer = model.layers[0]
    parameters = tuple(layer.parameters())

    def unexpected(*args, **kwargs):
        pytest.fail("Reusing a validated base must not read JSON or requantize weights")

    monkeypatch.setattr("unitunner.artifacts.quantization.tensor_json", unexpected)
    monkeypatch.setattr("unitunner.models.components.weight_quantization.quantize_weight", unexpected)
    assert quantize_model(model, precision, family="z-image") == 1
    assert model.layers[0] is layer
    assert all(left is right for left, right in zip(parameters, layer.parameters(), strict=True))
    different = "int8_rowwise" if precision != "int8_rowwise" else "int8_tensorwise"
    with pytest.raises(ValueError, match="existing quantization differs"):
        quantize_model(model, different, family="z-image")


@pytest.mark.parametrize("precision", (*FORMATS_TO_TEST, "fp8", "convrot-int8"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_text_base_online_and_offline_preparation_are_equivalent(precision, dtype):
    kitchen = pytest.importorskip("comfy_kitchen")

    def build():
        model = nn.Module()
        model.layers = nn.ModuleList([nn.Linear(256, 256, dtype=dtype) for _ in range(3)])
        return model

    model, restored = build(), build()
    config = TextEncodingConfig(quantization=precision)
    device = torch.device("cpu")
    with kitchen.use_backend("eager"):
        prepare_text_model(model, config, family="qwen3", device=device)
        load_text_state(restored, model.state_dict(), dtype=dtype, label="text base")
        prepare_text_model(restored, config, family="qwen3", device=device)
        values = torch.randn(2, 256, dtype=dtype)
        torch.testing.assert_close(model.layers[1](values), restored.layers[1](values), atol=0, rtol=0)
    assert isinstance(restored.layers[1], QuantizedLinear)
    assert not isinstance(restored.layers[0], QuantizedLinear)
    assert not isinstance(restored.layers[2], QuantizedLinear)
    assert all(not parameter.requires_grad for parameter in restored.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_base_quantizes_one_layer_at_a_time_on_gpu(monkeypatch):
    from unitunner.models.components import weight_quantization as implementation

    kitchen = pytest.importorskip("comfy_kitchen")
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(256, 128) for _ in range(3)])
    model.time_in = nn.Linear(256, 128)
    original = model.time_in.weight.detach().clone()
    seen = []
    quantizer = implementation.quantize_weight

    def checked(values, precision, **options):
        assert values.device.type == "cuda"
        # Previously completed projections return to CPU; future projections
        # remain floating on CPU rather than accumulating a GPU floating base.
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        seen.append(values.shape)
        return quantizer(values, precision, **options)

    monkeypatch.setattr(implementation, "quantize_weight", checked)
    with kitchen.use_backend("eager"):
        assert quantize_model(model, "float8_e4m3fn", family="z-image", device=torch.device("cuda")) == 3
    assert len(seen) == 3
    assert all(isinstance(layer, QuantizedLinear) for layer in model.layers)
    assert torch.equal(model.time_in.weight, original)


@pytest.mark.parametrize("precision", [name for name in FORMATS_TO_TEST if FORMATS[name].uses_rotation])
def test_existing_base_rejects_mismatched_quantizer_parameters(precision):
    kitchen = pytest.importorskip("comfy_kitchen")
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(256, 128)])
    with kitchen.use_backend("eager"):
        quantize_model(model, precision, family="z-image")
    layer = model.layers[0]
    before = tuple(layer.parameters())
    with pytest.raises(ValueError, match="rotation size differs"):
        quantize_model(model, precision, family="z-image", rotation_size=128)
    if FORMATS[precision].uses_group_size:
        with pytest.raises(ValueError, match="group size differs"):
            quantize_model(model, precision, family="z-image", group_size=32)
    assert all(a is b for a, b in zip(before, layer.parameters(), strict=True))
