"""Exercise real Kitchen layouts through file conversion and reload."""

import json
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from unitunner.artifacts.conversion import convert_weights
from unitunner.artifacts.formats.quantization import FORMATS
from unitunner.artifacts.quantization import read_weight, serialize_weight, tensor_json
from unitunner.artifacts.tensors import TensorReader, save_tensors
from unitunner.cli.main import main

DIRECT = tuple(name for name, spec in FORMATS.items() if not spec.calibrated)
DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


@pytest.fixture
def kitchen():
    module = pytest.importorskip("comfy_kitchen")
    pytest.importorskip("comfy_kitchen.tensor.awq_w4a16")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield module
    torch.set_num_threads(previous)


@pytest.mark.parametrize("precision", ("fp32", "fp16", "bf16"))
def test_float_cast_cli_preserves_metadata_integer_buffers_and_sources(tmp_path, precision, capsys):
    source, output = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    tensors = {"layer.weight": torch.randn(3, 5), "layer.bias": torch.randn(3), "steps": torch.arange(2)}
    save_tensors(tensors, source, metadata={"model": "test"})
    before = source.read_bytes()
    assert main(["weights", "convert", str(source), str(output), "--precision", precision]) == 0
    assert json.loads(capsys.readouterr().out)["converted"] == 2
    with TensorReader(output) as reader:
        assert reader.metadata()["model"] == "test"
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        for key, value in tensors.items():
            expected = value.to(dtype) if value.is_floating_point() else value
            torch.testing.assert_close(reader.read(key), expected, atol=0, rtol=0)
    assert source.read_bytes() == before


@pytest.mark.parametrize("precision", DIRECT)
@pytest.mark.parametrize("device", DEVICES)
def test_real_layout_file_roundtrip_and_float_export(kitchen, tmp_path, precision, device):
    from comfy_kitchen.tensor import QuantizedTensor

    torch.manual_seed(42)
    values = torch.randn(32, 256, dtype=torch.bfloat16)
    source, packed, restored = (
        tmp_path / name for name in ("source.safetensors", "packed.safetensors", "fp32.safetensors")
    )
    save_tensors({"linear.weight": values, "linear.bias": torch.zeros(32), "ids": torch.arange(2)}, source)
    original = source.read_bytes()
    report = convert_weights(source, packed, precision=precision, device=device, backend="eager")
    assert report["converted"] == 1
    with TensorReader(packed) as reader:
        config = tensor_json(reader.read("linear.comfy_quant"))
        weight = read_weight("linear.weight", reader.read, config, device=torch.device(device))
        with kitchen.use_backend("eager"):
            expected = weight.dequantize().float().cpu()
        # Independent public quantizer and direct layout dispatch validate the
        # converter's scale conventions, packing and retained constructor options.
        spec = FORMATS[precision]
        options = dict(spec.options)
        if precision.startswith("float8_"):
            options.update(
                dtype=getattr(torch, precision),
                scale=(values.float().abs().amax() / torch.finfo(getattr(torch, precision)).max).to(device),
            )
        elif precision == "nvfp4":
            options["scale"] = (values.float().abs().amax() / (448 * 6)).to(device)
        with kitchen.use_backend("eager"):
            reference_values = values.float() if precision == "int8_convrot" else values
            reference = QuantizedTensor.from_float(reference_values.to(device), spec.layout, **options)
            if precision == "int8_convrot":
                reference = QuantizedTensor(
                    reference._qdata, spec.layout, replace(reference._params, orig_dtype=values.dtype)
                )
            for key, value in reference.state_dict("linear.weight").items():
                actual = reader.read(key)
                # FP8 scales may be stored as bytes; compare the raw bits.
                assert torch.equal(
                    actual.reshape(-1).contiguous().view(torch.uint8),
                    value.cpu().reshape(-1).contiguous().view(torch.uint8),
                )
            x = torch.randn(3, 256, device=device, dtype=torch.bfloat16)
            torch.testing.assert_close(F.linear(x, weight), F.linear(x, reference), atol=0, rtol=0)
    convert_weights(packed, restored, precision="fp32", device=device, backend="eager")
    with TensorReader(restored) as reader:
        assert reader.keys() == ("ids", "linear.bias", "linear.weight")
        torch.testing.assert_close(reader.read("linear.weight"), expected, atol=0, rtol=0)
    assert source.read_bytes() == original


@pytest.mark.parametrize("precision", ("mxfp8", "nvfp4", "float8_e4m3fn", "float8_e5m2"))
def test_padding_keeps_logical_shape_and_zero_fp16_is_finite(kitchen, tmp_path, precision):
    source, output, dense = (
        tmp_path / name for name in ("input.safetensors", "packed.safetensors", "dense.safetensors")
    )
    save_tensors({"x.weight": torch.zeros(19, 67, dtype=torch.float16)}, source)
    convert_weights(source, output, precision=precision, backend="eager")
    convert_weights(output, dense, precision="fp32", backend="eager")
    with TensorReader(dense) as reader:
        weight = reader.read("x.weight")
        assert weight.shape == (19, 67)
        assert weight.count_nonzero() == 0


def test_shards_filters_and_atomic_failure(kitchen, tmp_path):
    first, second = (tmp_path / f"model-{n:02}-of-02.safetensors" for n in (1, 2))
    output = tmp_path / "result.safetensors"
    save_tensors({"a.weight": torch.randn(32, 256)}, first)
    save_tensors({"b.weight": torch.randn(3, 7)}, second)
    save_tensors({"old": torch.ones(1)}, output)
    before = output.read_bytes()
    with pytest.raises(ValueError, match="quantize"):
        convert_weights(second, output, precision="convrot_w4a4", backend="eager")
    assert output.read_bytes() == before
    report = convert_weights(second, output, precision="convrot_w4a4", backend="eager", on_incompatible="keep")
    assert report["converted"] == 1 and "b.weight" in report["skipped"]
    convert_weights(first, output, precision="int8_tensorwise", exclude=("b.*",), backend="eager")
    with TensorReader(output) as reader:
        assert reader.read("b.weight").shape == (3, 7)
        assert "b.comfy_quant" not in reader.keys()
    with pytest.raises(ValueError, match="destination"):
        convert_weights(second, first, precision="fp32")
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("precision,tiled", (("awq_w4a16", False), ("svdquant_w4a4", False), ("svdquant_w4a4", True)))
def test_calibrated_import_preservation_and_dequantization(kitchen, tmp_path, precision, tiled):
    from comfy_kitchen import tensor as kt

    torch.manual_seed(3)
    n, k, rank = 128 if tiled else 32, 128, 8
    packed = torch.randint(-128, 128, (n, k // 2), dtype=torch.int8)
    scale = torch.rand(k // 64, n, dtype=torch.bfloat16)
    if precision == "awq_w4a16":
        layout = kt.TensorCoreAWQW4A16Layout
        params = layout.Params(scale=scale, zeros=torch.zeros_like(scale), orig_dtype=torch.bfloat16, orig_shape=(n, k))
    else:
        layout = kt.TensorCoreSVDQuantW4A4Layout
        up = torch.randn(n, rank, dtype=torch.bfloat16) * 0.01
        if tiled:
            packed = (
                packed.view(n // 128, 32, 4, k // 64, 32).permute(0, 3, 1, 2, 4).reshape(n // 128, k // 64, 32, 128)
            )
            scale = scale.view(k // 64, n // 128, 128).permute(1, 0, 2).contiguous()
            up = up.view(n // 128, 128, rank).transpose(1, 2).contiguous()
        params = layout.Params(
            scale=scale,
            proj_down=torch.randn(k, rank, dtype=torch.bfloat16) * 0.01,
            proj_up=up,
            smooth_factor=torch.ones(k, dtype=torch.bfloat16),
            orig_dtype=torch.bfloat16,
            orig_shape=(n, k),
        )
    reference = kt.QuantizedTensor(packed, layout.__name__, params)
    tensors = serialize_weight("x.weight", reference, precision)
    spec = tmp_path / "layout.json"
    spec.write_text(json.dumps({"x": tensor_json(tensors.pop("x.comfy_quant"))}))
    source, output, dense = (
        tmp_path / name for name in ("calibrated.safetensors", "model.safetensors", "dense.safetensors")
    )
    save_tensors(tensors, source)
    report = convert_weights(source, output, precision=precision, layout_spec=spec, backend="eager")
    assert report["preserved_quantized"] == 1
    convert_weights(output, dense, precision="fp32", backend="eager")
    with kitchen.use_backend("eager"), TensorReader(dense) as reader:
        torch.testing.assert_close(reader.read("x.weight"), reference.dequantize().float(), atol=0, rtol=0)
    save_tensors({"x.weight": torch.randn(n, k)}, source)
    before = output.read_bytes()
    with pytest.raises(ValueError, match="calibrat"):
        convert_weights(source, output, precision=precision)
    assert output.read_bytes() == before


def test_rejects_missing_or_invalid_sidecars_and_folds_smoothing(kitchen, tmp_path):
    source, output = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    save_tensors({"x.weight": torch.randn(3, 8)}, source)
    convert_weights(source, output, precision="float8_e4m3fn", backend="eager")
    with TensorReader(output) as reader:
        data = {key: reader.read(key) for key in reader.keys()}
        config = tensor_json(data["x.comfy_quant"])
        expected = read_weight("x.weight", data.__getitem__, config, device=torch.device("cpu")).dequantize().float()
    data["x.pre_quant_scale"] = torch.arange(1, 9).float()
    save_tensors(data, source)
    convert_weights(source, output, precision="fp32")
    with TensorReader(output) as reader:
        torch.testing.assert_close(reader.read("x.weight"), expected * torch.arange(1, 9), atol=0, rtol=0)
        assert "x.pre_quant_scale" not in reader.keys()
    del data["x.weight_scale"]
    save_tensors(data, source)
    before = output.read_bytes()
    with pytest.raises(ValueError, match="Cannot load"):
        convert_weights(source, output, precision="keep")
    assert output.read_bytes() == before


def test_unrecognized_integer_weight_is_not_silently_cast(tmp_path):
    source, output = tmp_path / "source.safetensors", tmp_path / "output.safetensors"
    save_tensors({"x.weight": torch.ones(4, 4, dtype=torch.int8)}, source)
    with pytest.raises(ValueError, match="layout-spec"):
        convert_weights(source, output, precision="bf16")
    assert not output.exists()


def test_w6_reload_and_excluded_layers_preserve_sidecars(kitchen, tmp_path):
    source, packed, output = (
        tmp_path / name for name in ("source.safetensors", "packed.safetensors", "output.safetensors")
    )
    save_tensors({"x.weight": torch.randn(32, 256, dtype=torch.bfloat16)}, source)
    convert_weights(source, packed, precision="w6a8_int8", backend="eager")
    with TensorReader(packed) as reader:
        tensors = {key: reader.read(key) for key in reader.keys()}
        config = tensor_json(tensors["x.comfy_quant"])
        expected = read_weight("x.weight", reader.read, config, device=torch.device("cpu")).dequantize()
    from unitunner.artifacts.quantization import json_tensor

    tensors["x.comfy_quant"] = json_tensor(config)
    tensors["x.input_scale"] = torch.tensor(0.5)
    save_tensors(tensors, packed)
    convert_weights(packed, output, precision="bf16", exclude=("x.*",), backend="eager")
    with TensorReader(output) as reader:
        for key, tensor in tensors.items():
            if key != "x.comfy_quant":
                assert torch.equal(reader.read(key).reshape(-1).view(torch.uint8), tensor.reshape(-1).view(torch.uint8))
        assert tensor_json(reader.read("x.comfy_quant"))["format"] == "w6a8_int8"
    convert_weights(packed, output, precision="bf16", backend="eager")
    with TensorReader(output) as reader:
        torch.testing.assert_close(reader.read("x.weight"), expected, atol=0, rtol=0)
        assert reader.keys() == ("x.weight",)
