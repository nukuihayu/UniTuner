"""Compare real tiny native models against official and fused ComfyUI reference layouts."""

import json
from importlib import import_module

import pytest
import torch

from tests.models.test_transformer_external_loading import _CASES
from unitunner.analysis.references import inspect_reference
from unitunner.analysis.run import analyze, baseline_key
from unitunner.artifacts.formats.transformer import convert_transformer, official_projection_sources
from unitunner.artifacts.tensors import save_tensors


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def official_state(family, native):
    """Build published-layout fixtures and check them against the actual model importer."""
    result = {}
    for name, value in native.items():
        paths = official_projection_sources(family, name)
        if len(paths) == 1:
            pieces = [value]
            if family == "minimax-h3" and name.endswith(".mlp.fc1.weight"):
                gate, up = value.chunk(2, dim=0)
                pieces = [torch.cat((up, gate))]
        elif len(paths) == 3:
            pieces = value.chunk(3, dim=0)
        else:
            width = value.shape[1] if value.ndim == 2 else native[name.removesuffix("bias") + "weight"].shape[1]
            pieces = value.split((width, width, width, value.shape[0] - 3 * width), dim=0)
        result.update(zip(paths, pieces, strict=True))
    return result


@pytest.mark.parametrize("module,stem,options", _CASES)
@pytest.mark.parametrize("layout", ["native", "official"])
@pytest.mark.parametrize("embedded", [False, True])
def test_all_model_projection_references(tmp_path, module, stem, options, layout, embedded):
    config = getattr(import_module(f"unitunner.models.{module}.config.model"), stem + "Config")(**options)
    model = getattr(import_module(f"unitunner.models.{module}.transformer"), stem + "Transformer")(config)
    targets = import_module(f"unitunner.models.{module}.projections").adapter_projections(model)
    family = module.replace("_", "-")
    native = model.state_dict()
    published = official_state(family, native) if layout == "official" else native
    if layout == "official":
        imported = convert_transformer(module, published, native, external=True)
        for name in targets:
            torch.testing.assert_close(imported[name + ".weight"], native[name + ".weight"], rtol=0, atol=0)
    path = tmp_path / "base.safetensors"
    save_tensors(
        {"model.diffusion_model." + k: v for k, v in published.items()},
        path,
        metadata={f"unitunner.{module}.config": config.model_dump_json()} if embedded else None,
    )
    with inspect_reference(path) as reference:
        for name in targets:
            key = baseline_key("weights/transformer." + name + ".weight", reference, {})
            assert key is not None, (family, layout, name)
            torch.testing.assert_close(reference.read(key), native[name + ".weight"], rtol=0, atol=0)
        # Official LoRA projections also resolve against native/fused references.
        for name in targets:
            for official in official_projection_sources(family, name + ".weight"):
                key = baseline_key("weights/transformer." + official, reference, {})
                assert key is not None, (family, layout, official)
                expected = official_state(family, {name + ".weight": native[name + ".weight"]})[official]
                torch.testing.assert_close(reference.read(key), expected, rtol=0, atol=0)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("indexed", [False, True])
def test_qwen21_fused_reference_report_matches_official(tmp_path, wrapped, indexed):
    torch.manual_seed(42)
    prefix = "model.diffusion_model." if wrapped else ""
    gate, proj = torch.randn(12, 4), torch.randn(12, 4) * 7
    official, fused = tmp_path / "official.safetensors", tmp_path / "fused.safetensors"
    root = "transformer_blocks.0.img_mlp."
    save_tensors({root + "gate_layer.weight": gate, root + "proj.weight": proj}, official)
    save_tensors({prefix + root + "gate_up.weight": torch.cat((gate, proj))}, fused)
    if indexed:
        index = tmp_path / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {prefix + root + "gate_up.weight": fused.name}}))
        fused = index
    adapter = tmp_path / "adapter.safetensors"
    tensors = {}
    for name in ("gate_layer", "proj"):
        local = "transformer." + root + name
        tensors.update(
            {
                local + ".lora_down.weight": torch.randn(2, 4),
                local + ".lora_up.weight": torch.randn(12, 2),
                local + ".alpha": torch.tensor(3.0),
            }
        )
    save_tensors(tensors, adapter)
    a, b = (analyze(adapter, base=base, top_layers=2, top_k=2) for base in (official, fused))
    assert a["snapshots"][0]["coverage"]["base_matched"] == 2
    assert b["snapshots"][0]["coverage"]["base_matched"] == 2
    for left, right in zip(a["snapshots"][0]["effective"], b["snapshots"][0]["effective"], strict=True):
        assert left["base"]["relative_delta"] == pytest.approx(right["base"]["relative_delta"])
    for left, right in zip(a["subspaces"], b["subspaces"], strict=True):
        assert left["status"] == right["status"] == "available"
        assert left["pairs"][0]["subspace_overlap"] == pytest.approx(right["pairs"][0]["subspace_overlap"])


def test_z_image_gqa_and_flux_single_fused_rows(tmp_path):
    z = torch.arange(48.0).reshape(8, 6)  # Q=6, K=1, V=1, not equal thirds.
    flux = torch.arange(80.0).reshape(20, 4)
    path = tmp_path / "base.safetensors"
    save_tensors({"layers.0.attention.qkv.weight": z, "single_blocks.0.linear1.weight": flux}, path)
    with inspect_reference(path) as artifact:
        for name, expected in [
            ("layers.0.attention.to_q.weight", z[:6]),
            ("layers.0.attention.to_k.weight", z[6:7]),
            ("layers.0.attention.to_v.weight", z[7:]),
            ("single_transformer_blocks.0.attn.to_k.weight", flux[4:8]),
            ("single_transformer_blocks.0.proj_mlp.weight", flux[12:]),
        ]:
            key = baseline_key("weights/" + name, artifact, {})
            assert key is not None
            torch.testing.assert_close(artifact.read(key), expected, rtol=0, atol=0)


def test_quantized_and_ambiguous_references_stay_unavailable(tmp_path):
    path = tmp_path / "base.safetensors"
    save_tensors(
        {
            "transformer_blocks.0.img_mlp.gate_up.weight": torch.ones(8, 4, dtype=torch.int8),
            "transformer_blocks.0.img_mlp.gate_up.comfy_quant": torch.zeros(2, dtype=torch.uint8),
            "transformer.x.weight": torch.ones(4, 4),
            "diffusion_model.x.weight": torch.ones(4, 4) * 2,
        },
        path,
    )
    with inspect_reference(path) as artifact:
        key = baseline_key("weights/transformer_blocks.0.img_mlp.proj.weight", artifact, {})
        assert key in artifact.storage_only
        assert baseline_key("weights/x.weight", artifact, {}) is None


def test_qwen21_all_224_projections_match_fused_base(tmp_path):
    base, lora = {}, {}
    torch.manual_seed(31)
    for block in range(32):
        prefix = f"transformer_blocks.{block}."
        for projection, inputs, outputs in [
            ("attn.to_q", 4, 4),
            ("attn.to_k", 4, 4),
            ("attn.to_v", 4, 4),
            ("attn.to_out.0", 4, 4),
            ("img_mlp.gate_layer", 4, 12),
            ("img_mlp.proj", 4, 12),
            ("img_mlp.out", 12, 4),
        ]:
            path = prefix + projection
            base[path + ".weight"] = torch.randn(outputs, inputs)
            lora["transformer." + path + ".lora_down.weight"] = torch.randn(2, inputs)
            lora["transformer." + path + ".lora_up.weight"] = torch.randn(outputs, 2)
            lora["transformer." + path + ".alpha"] = torch.tensor(3.0)
        base[prefix + "img_mlp.gate_up.weight"] = torch.cat(
            [base.pop(prefix + "img_mlp.gate_layer.weight"), base.pop(prefix + "img_mlp.proj.weight")]
        )
    baseline, adapter = tmp_path / "base.safetensors", tmp_path / "lora.safetensors"
    save_tensors(base, baseline)
    save_tensors(lora, adapter)
    report = analyze(adapter, base=baseline, top_layers=3)
    coverage = report["snapshots"][0]["coverage"]
    assert coverage["base_matched"] == 224
    assert coverage["base_unmatched"] == 0
