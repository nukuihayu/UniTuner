"""Exercise frozen Kitchen bases through native loading, adapter updates and replay."""

import importlib
from contextlib import closing

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.assertions import require_type, require_value
from unituner.adapters.config import AdapterConfig
from unituner.artifacts.conversion import convert_weights
from unituner.artifacts.formats.quantization import FORMATS
from unituner.artifacts.quantization import quantize_weight, serialize_weight, tensor_json
from unituner.artifacts.quantized_loading import load_quantized_model
from unituner.artifacts.tensors import TensorReader
from unituner.models.components.quantized import QuantizedLinear
from unituner.training.parameters import ParameterInstallation


@pytest.fixture(autouse=True)
def kitchen():
    module = pytest.importorskip("comfy_kitchen")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with module.use_backend("eager"):
        yield module
    torch.set_num_threads(previous)


DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


@pytest.mark.parametrize("precision", ["awq_w4a16", "svdquant_w4a4"])
def test_calibrated_canonical_bases_support_adapter_gradients(kitchen, precision):
    from comfy_kitchen import tensor as kt

    n, k, rank = 32, 128, 8
    packed = torch.randint(-128, 128, (n, k // 2), dtype=torch.int8)
    scale = torch.rand(k // 64, n, dtype=torch.bfloat16)
    if precision == "awq_w4a16":
        layout = kt.TensorCoreAWQW4A16Layout
        params = layout.Params(scale=scale, zeros=torch.zeros_like(scale), orig_dtype=torch.bfloat16, orig_shape=(n, k))
    else:
        layout = kt.TensorCoreSVDQuantW4A4Layout
        params = layout.Params(
            scale=scale,
            proj_down=torch.randn(k, rank, dtype=torch.bfloat16) * 0.01,
            proj_up=torch.randn(n, rank, dtype=torch.bfloat16) * 0.01,
            smooth_factor=torch.ones(k, dtype=torch.bfloat16),
            orig_dtype=torch.bfloat16,
            orig_shape=(n, k),
        )
    reference = kt.QuantizedTensor(packed, layout.__name__, params)
    tensors = serialize_weight("0.weight", reference, precision)
    model = nn.Sequential(nn.Linear(k, n, bias=False))
    assert load_quantized_model(model, tensors, device=torch.device("cpu"), dtype=torch.bfloat16)
    assert isinstance(model[0], QuantizedLinear)
    values = torch.randn(2, k, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model(values)
        expected = F.linear(values, reference.dequantize())
    torch.testing.assert_close(actual, expected)
    actual.float().square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def regular_hadamard(size, device, dtype):
    # Independent regular H4, not the Sylvester matrix: ConvRot preserves constants.
    h = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], device=device, dtype=dtype)
    while h.shape[0] < size:
        h = torch.kron(h, h.new_tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]))
    return h / size**0.5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_convrot_fp32_codes_and_floating_weight_gradients(device, dtype):
    torch.manual_seed(61)
    values = torch.randn(64, 256, dtype=dtype, device=device)
    weight = quantize_weight(values, "int8_convrot")
    h = regular_hadamard(256, device, torch.float32)
    rotated = values.float().reshape(64, 1, 256) @ h.T
    rotated = rotated.reshape(64, 256)
    scales = (rotated.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-30)
    codes = (rotated / scales).round().clamp(-127, 127).to(torch.int8)
    assert torch.equal(codes, weight._qdata)
    torch.testing.assert_close(scales, weight._params.scale, atol=0, rtol=0)
    bias = torch.randn(64, dtype=dtype, device=device)
    state = serialize_weight("weight", weight, "int8_convrot")
    state["bias"] = bias
    layer = QuantizedLinear(state, dtype=dtype)
    inputs = torch.randn(2, 3, 256, dtype=dtype, device=device, requires_grad=True)
    upstream = torch.randn(2, 3, 64, dtype=dtype, device=device)
    output = layer(inputs)
    output.backward(upstream)
    dense_weight = weight.dequantize().to(dtype)
    torch.testing.assert_close(inputs.grad, upstream @ dense_weight, atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in layer.parameters())
    torch.testing.assert_close(output, F.linear(inputs, dense_weight, bias), atol=0, rtol=0)
    dense = F.linear(inputs.detach().float(), values.float(), bias.float())
    assert (output.float() - dense).norm() / dense.norm() < 0.025


CASES = {
    "flux": (
        "Flux",
        {
            "in_channels": 8,
            "out_channels": 8,
            "vec_in_dim": 6,
            "context_in_dim": 12,
            "hidden_size": 32,
            "num_heads": 2,
            "depth": 1,
            "depth_single_blocks": 1,
            "axes_dim": (4, 6, 6),
            "mlp_ratio": 2.0,
        },
    ),
    "flux2": (
        "Flux2",
        {
            "in_channels": 8,
            "context_in_dim": 12,
            "hidden_size": 32,
            "num_heads": 2,
            "depth": 1,
            "depth_single_blocks": 1,
            "axes_dim": (4, 4, 4, 4),
        },
    ),
    "flux3": (
        "Flux3",
        {
            "hidden_size": 32,
            "num_heads": 2,
            "depth": 1,
            "depth_single": 1,
            "axes_dim": (4, 4, 4, 4),
            "image_channels": 8,
            "text_channels": 12,
            "mlp_ratio": 2.0,
        },
    ),
    "z_image": (
        "ZImage",
        {
            "in_channels": 2,
            "dim": 24,
            "num_layers": 2,
            "num_refiner_layers": 1,
            "num_heads": 4,
            "num_kv_heads": 2,
            "caption_dim": 8,
            "timestep_hidden_dim": 32,
            "axes_dims": (2, 2, 2),
            "axes_lengths": (128, 32, 32),
        },
    ),
    "qwen_image": (
        "QwenImage",
        {
            "in_channels": 8,
            "out_channels": 2,
            "num_layers": 2,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "joint_attention_dim": 12,
            "axes_dims_rope": (4, 6, 6),
        },
    ),
    "qwen_image_2_1": (
        "QwenImage21",
        {
            "in_channels": 8,
            "out_channels": 8,
            "num_layers": 2,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "context_in_dim": 12,
            "axes_dims_rope": (4, 6, 6),
        },
    ),
    "moss_image": (
        "MossImage",
        {
            "dim": 32,
            "ffn_dim": 48,
            "num_heads": 2,
            "num_layers": 2,
            "text_dim": 12,
            "freq_dim": 8,
            "latent_channels": 8,
        },
    ),
    "krea2": (
        "Krea2",
        {
            "hidden_size": 64,
            "timestep_dim": 16,
            "text_dim": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "mlp_multiplier": 2,
            "num_layers": 1,
            "text_layers": 3,
            "text_heads": 4,
            "text_kv_heads": 2,
        },
    ),
    "wan": (
        "Wan",
        {
            "hidden_size": 48,
            "num_heads": 4,
            "num_layers": 2,
            "ffn_dim": 80,
            "timestep_dim": 16,
            "text_dim": 32,
            "text_length": 7,
        },
    ),
    "minimax_h3": (
        "MiniMaxH3",
        {
            "hidden_size": 24,
            "num_layers": 2,
            "token_refiner_num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "ffn_hidden_size": 40,
            "text_dim": 12,
            "timestep_input_dim": 8,
            "time_embed_hidden_size": 16,
            "time_embed_dim": 8,
            "rope_inv_freq_len": 2,
        },
    ),
}


@pytest.mark.parametrize("family", CASES)
@pytest.mark.parametrize("algorithm", ("lora", "loha", "lokr"))
def test_all_family_load_adapter_update_and_quantized_export(tmp_path, family, algorithm):
    classname, fields = CASES[family]
    config = getattr(importlib.import_module(f"unituner.models.{family}.config.model"), classname + "Config")(**fields)
    model = getattr(importlib.import_module(f"unituner.models.{family}.transformer"), classname + "Transformer")(
        config
    )
    api = importlib.import_module(f"unituner.models.{family}.loading")
    suffix = "qwen_image_21" if family == "qwen_image_2_1" else family
    save, load = getattr(api, f"save_{suffix}"), getattr(api, f"load_{suffix}")
    source, packed, republished = (
        tmp_path / name for name in ("float.safetensors", "int8.safetensors", "republished.safetensors")
    )
    save(model, source)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    report = convert_weights(
        source, packed, precision="int8_convrot", rotation_size=4, on_incompatible="keep", backend="eager"
    )
    assert report["converted"] > 0
    rng = torch.get_rng_state().clone()
    restored = load(packed)
    assert torch.equal(rng, torch.get_rng_state())
    quantized = {name: module for name, module in restored.named_modules() if isinstance(module, QuantizedLinear)}
    assert len(quantized) == report["converted"]
    for key, reason in report["skipped"].items():
        assert reason
        torch.testing.assert_close(restored.state_dict()[key], original[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="adapter"):
        ParameterInstallation(restored)
    targets = importlib.import_module(f"unituner.models.{family}.projections").adapter_projections(restored)
    selected = {name: module for name, module in targets.items() if name in quantized}
    assert selected
    before = {name: layer.weight.clone() for name, layer in quantized.items()}
    with closing(
        ParameterInstallation(restored, adapter=AdapterConfig(algorithm=algorithm, rank=2, alpha=2), targets=selected)
    ) as owner:
        optimizer = torch.optim.AdamW(owner.plan.parameters, lr=0.01)
        initial = [parameter.detach().clone() for parameter in owner.plan.parameters]
        loss = require_type(
            sum(layer(torch.randn(2, layer.in_features)).square().mean() for layer in selected.values()), torch.Tensor
        )
        loss.backward()
        gradients = [parameter.grad for parameter in owner.plan.parameters]
        assert all(value is not None and torch.isfinite(value).all() for value in gradients)
        assert any(require_value(value).count_nonzero() for value in gradients)
        optimizer.step()
        assert any(
            not torch.equal(parameter, old) for parameter, old in zip(owner.plan.parameters, initial, strict=True)
        )
        assert all(
            torch.equal(layer.weight, before[name]) and layer.weight.grad is None for name, layer in quantized.items()
        )
    # Publication casts dense weights, retaining scale precision and wire metadata.
    save(restored, republished, dtype=torch.bfloat16)
    reloaded = load(republished)
    for name, layer in quantized.items():
        other = reloaded.get_submodule(name)
        assert other.weight.dtype == torch.int8 and other.get_parameter("weight_scale").dtype == torch.float32
        assert torch.equal(layer.get_parameter("weight_scale"), other.get_parameter("weight_scale"))
        assert torch.equal(layer.weight, other.weight)
        if family == "z_image":
            assert other.compute_dtype == torch.bfloat16
            assert other(torch.randn(2, other.in_features, dtype=torch.bfloat16)).dtype == torch.bfloat16
    with TensorReader(republished) as reader:
        first = next(iter(quantized))
        assert tensor_json(reader.read(first + ".comfy_quant"))["convrot"] is True


def test_quantized_linear_dtype_move_preserves_fp32_scales():
    values = torch.randn(16, 16)
    weight = quantize_weight(values, "int8_convrot", rotation_size=4)
    layer = QuantizedLinear(serialize_weight("weight", weight, "int8_convrot"), dtype=torch.float32)
    original = layer.get_parameter("weight_scale").clone()
    layer.bfloat16()
    assert layer.compute_dtype == torch.bfloat16
    assert layer.get_parameter("weight_scale").dtype == torch.float32
    assert torch.equal(layer.get_parameter("weight_scale"), original)


@pytest.mark.parametrize("precision", tuple(name for name, spec in FORMATS.items() if not spec.calibrated))
@pytest.mark.parametrize("compiled", (False, True))
def test_cuda_frozen_block_replay_and_amp_match_resident(compiled, precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from contextlib import ExitStack
    from copy import deepcopy

    from unituner.execution.compiler import CompilerInstallation
    from unituner.execution.frozen_blocks import FrozenBlockInstallation
    from unituner.execution.regions import ModelRegion

    torch.manual_seed(910)
    device = torch.device("cuda", torch.cuda.current_device())
    layers = []
    for _ in range(2):
        packed = quantize_weight(torch.randn(64, 64, device=device), precision, rotation_size=64)
        layers.append(QuantizedLinear(serialize_weight("weight", packed, precision), dtype=torch.float32))
    resident = nn.Sequential(layers[0], nn.SiLU(), layers[1])
    staged = deepcopy(resident)
    with ExitStack() as owners:
        installations = []
        for model in (resident, staged):
            installations.append(
                owners.enter_context(
                    closing(
                        ParameterInstallation(
                            model,
                            adapter=AdapterConfig(rank=2),
                            targets={
                                "0": require_type(model[0], torch.nn.Linear),
                                "2": require_type(model[2], torch.nn.Linear),
                            },
                        )
                    )
                )
            )
        installations[1].branches.load_state_dict(installations[0].branches.state_dict())
        if compiled:
            for model in (resident, staged):
                owners.enter_context(closing(CompilerInstallation(model, backend="eager", precision="bf16")))
        owners.enter_context(closing(FrozenBlockInstallation(ModelRegion(staged, 0, "visual"), device=device)))
        codes = [layer.weight.clone() for layer in resident if isinstance(layer, QuantizedLinear)]
        optimizers = [torch.optim.AdamW(owner.plan.parameters, lr=0.001) for owner in installations]
        for _ in range(2):
            inputs = torch.randn(2, 3, 64, device=device)
            outputs, input_gradients = [], []
            for model, owner, optimizer in zip((resident, staged), installations, optimizers, strict=True):
                x = inputs.clone().requires_grad_()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(x)
                output.float().square().mean().backward()
                assert x.grad is not None
                assert x.grad.dtype == torch.float32 and torch.isfinite(x.grad).all()
                outputs.append(output.detach())
                input_gradients.append(x.grad)
                assert all(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    for parameter in owner.plan.parameters
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
            torch.testing.assert_close(input_gradients[0], input_gradients[1], atol=0, rtol=0)
            for left, right in zip(installations[0].plan.parameters, installations[1].plan.parameters, strict=True):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        assert all(
            torch.equal(layer.get_parameter("weight").view(torch.uint8), code.view(torch.uint8))
            for layer, code in zip((resident[0], resident[2]), codes, strict=True)
        )
