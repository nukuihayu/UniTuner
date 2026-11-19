"""Check bounded image loading against the original FP32 quantization path."""

import json
from importlib import import_module

import pytest
import torch

from tests.models.test_transformer_external_loading import _CASES
from unituner.artifacts.formats.transformer import convert_transformer
from unituner.artifacts.quantization import quantize_weight, serialize_weight
from unituner.config.execution import ExecutionConfig
from unituner.execution.installation import prepare_base_weights


@pytest.mark.parametrize("module,stem,options", [case for case in _CASES if case[0] != "wan"])
@pytest.mark.parametrize("precision", ["float8_e4m3fn", "int8_rowwise"])
def test_source_precision_loading_matches_fp32_quantization(tmp_path, module, stem, options, precision):
    kitchen = pytest.importorskip("comfy_kitchen")
    config = getattr(import_module(f"unituner.models.{module}.config.model"), stem + "Config")(**options)
    model = getattr(import_module(f"unituner.models.{module}.transformer"), stem + "Transformer")(config)
    if "rope.inv_freq" in model.state_dict():
        model.state_dict()["rope.inv_freq"].fill_(0.25)
    model.to(dtype=torch.bfloat16)
    loading = import_module(f"unituner.models.{module}.loading")
    suffix = "qwen_image_21" if module == "qwen_image_2_1" else module
    path = tmp_path / "source.safetensors"
    getattr(loading, "save_" + suffix)(model, path)
    loader = getattr(loading, "load_" + suffix)
    reference = loader(path, dtype=torch.float32)
    optimized = loader(path, dtype=torch.float32, preserve_quantization_sources=True)
    reference_bytes = sum(value.numel() * value.element_size() for value in reference.state_dict().values())
    optimized_bytes = sum(value.numel() * value.element_size() for value in optimized.state_dict().values())
    assert optimized_bytes < reference_bytes
    execution = ExecutionConfig(precision="bf16", quantization=precision)
    family = module.replace("_", "-")
    with kitchen.use_backend("eager"):
        prepare_base_weights(reference, execution, family=family, device=torch.device("cpu"))
        prepare_base_weights(optimized, execution, family=family, device=torch.device("cpu"))
    assert reference.state_dict().keys() == optimized.state_dict().keys()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(optimized.state_dict()[key].float(), value.float(), atol=0, rtol=0)


@pytest.mark.parametrize("precision", ["float8_e4m3fn", "int8_rowwise"])
def test_h3_image_source_precision_preserves_pdd_and_quantization(tmp_path, precision):
    kitchen = pytest.importorskip("comfy_kitchen")
    from tests.models.test_minimax_h3_image import _config
    from unituner.models.minimax_h3.image import (
        MiniMaxH3ImageTransformer,
        load_image_transformer,
        save_image_transformer,
    )

    model = MiniMaxH3ImageTransformer(_config(), pdd_sigmas=(1.0, 0.0), dtype=torch.float32)
    model.to(dtype=torch.bfloat16)
    path, schedule = tmp_path / "image.safetensors", tmp_path / "schedule.json"
    schedule.write_text(json.dumps({"num_inference_steps": 1, "sigmas": [1.0, 0.0]}))
    save_image_transformer(model, path)
    arguments = {"config": _config(), "pdd_config": schedule, "dtype": torch.float32}
    reference = load_image_transformer(path, **arguments)
    optimized = load_image_transformer(path, preserve_quantization_sources=True, **arguments)
    assert optimized.blocks[0].attn.qkv_proj.weight.dtype == torch.bfloat16
    execution = ExecutionConfig(precision="bf16", quantization=precision)
    with kitchen.use_backend("eager"):
        for loaded in (reference, optimized):
            prepare_base_weights(loaded, execution, family="minimax-h3-image", device=torch.device("cpu"))
    assert reference.state_dict().keys() == optimized.state_dict().keys()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(optimized.state_dict()[key].float(), value.float(), atol=0, rtol=0)


@pytest.mark.parametrize(
    "family,native,external",
    [
        ("flux", "double_blocks.0.img_mlp.0.weight", "transformer_blocks.0.ff.net.0.proj.weight"),
        ("flux2", "double_blocks.0.img_mlp.0.weight", "transformer_blocks.0.ff.linear_in.weight"),
        ("flux3", "single_blocks.0.mlp_in.weight", "single_transformer_blocks.0.ff.linear_in.weight"),
        ("krea2", "blocks.0.attn.wq.weight", "transformer_blocks.0.attn.to_q.weight"),
        ("minimax_h3", "blocks.0.mlp.fc2.weight", "transformer_blocks.0.ff.net.2.weight"),
    ],
)
@pytest.mark.parametrize("precision", ["float8_e4m3fn", "int8_rowwise"])
def test_external_one_to_one_mapping_keeps_packed_storage(monkeypatch, family, native, external, precision):
    pytest.importorskip("comfy_kitchen")
    weight = torch.randn(16, 16)
    state = serialize_weight(external, quantize_weight(weight, precision), precision)
    expected_topology = {native: weight}
    if family == "flux3":
        state["context_embedder.weight"] = weight
        expected_topology["txt_in.weight"] = weight

    def reject_decode(*args, **kwargs):
        pytest.fail("One-to-one mapping must preserve packed storage")

    monkeypatch.setattr("unituner.artifacts.formats.transformer.dequantize_transformer", reject_decode)
    converted = convert_transformer(family, state, expected_topology, external=True)
    expected = serialize_weight(native, quantize_weight(weight, precision), precision)
    if family == "flux3":
        expected["txt_in.weight"] = weight
    assert converted.keys() == expected.keys()
    assert converted[native].data_ptr() == state[external].data_ptr()


@pytest.mark.parametrize("family", ["flux", "flux2", "minimax_h3"])
def test_external_fusion_decodes_only_required_projections(family):
    kitchen = pytest.importorskip("comfy_kitchen")
    from unituner.artifacts.formats.transformer import _flux_sources, _h3_sources

    fused = "blocks.0.attn.qkv_proj.weight" if family == "minimax_h3" else "double_blocks.0.img_attn.qkv.weight"
    untouched = "blocks.0.mlp.fc2.weight" if family == "minimax_h3" else "double_blocks.0.img_mlp.2.weight"
    mapper = _h3_sources if family == "minimax_h3" else lambda key: _flux_sources(key, second=family == "flux2")
    source, decoded = {}, []
    with kitchen.use_backend("eager"):
        for key in mapper(fused):
            packed = quantize_weight(torch.randn(16, 16), "int8_rowwise")
            source.update(serialize_weight(key, packed, "int8_rowwise"))
            decoded.append(packed.dequantize())
        external = mapper(untouched)[0]
        source.update(serialize_weight(external, quantize_weight(torch.randn(16, 16), "int8_rowwise"), "int8_rowwise"))
        converted = convert_transformer(
            family, source, {fused: torch.empty(48, 16), untouched: torch.empty(16, 16)}, external=True
        )
    torch.testing.assert_close(converted[fused], torch.cat(decoded), atol=0, rtol=0)
    assert converted[untouched].data_ptr() == source[external].data_ptr()
    assert untouched.removesuffix("weight") + "comfy_quant" in converted
