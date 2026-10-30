"""Check official LoRA factors against native updates without merging adapter layers."""

import json
from contextlib import closing
from importlib import import_module

import pytest
import torch

from tests.models.test_transformer_external_loading import _CASES
from unituner.adapters.config import AdapterConfig
from unituner.analysis.adapters import groups, load_delta
from unituner.analysis.artifacts import inspect_artifact
from unituner.artifacts.formats.transformer import convert_transformer
from unituner.artifacts.tensors import TensorReader, save_tensors
from unituner.training.parameters import ParameterInstallation
from unituner.training.publishing import publish_adapter


@pytest.mark.parametrize("module,stem,options", _CASES)
def test_official_factors_preserve_native_updates_and_source_state(tmp_path, module, stem, options):
    config = getattr(import_module(f"unituner.models.{module}.config.model"), stem + "Config")(**options)
    model = getattr(import_module(f"unituner.models.{module}.transformer"), stem + "Transformer")(config)
    targets = import_module(f"unituner.models.{module}.projections").adapter_projections(model)
    adapter = AdapterConfig(rank=2, alpha=7.0)
    with closing(ParameterInstallation(model, adapter=adapter, targets=targets)) as installation:
        with torch.no_grad():
            for value in installation.branches.parameters():
                value.normal_(0, 0.2)
        before = {k: v.clone() for k, v in installation.branches.state_dict().items()}
        identities = [id(p) for p in model.parameters()]
        path = tmp_path / "official.safetensors"
        publish_adapter(
            path,
            branches=installation.branches,
            target_names=installation.plan.target_names,
            adapter=adapter,
            base_model=module,
            module_identifier="networks.lora_test",
            dtype=torch.float32,
            adapter_format="diffusers",
            family=module.replace("_", "-"),
        )
        with TensorReader(path) as reader:
            assert reader.metadata()["unituner.adapter.format"] == "diffusers"
            assert all(key.endswith((".lora_A.weight", ".lora_B.weight")) for key in reader.keys())
        published_deltas = {}
        with inspect_artifact(path) as artifact:
            for name, parts in groups(artifact).items():
                delta = load_delta(artifact, name, parts)
                assert delta.rank == 2 and delta.scale == 1
                key = name.removeprefix("weights/transformer.") + ".weight"
                published_deltas[key] = delta.rows(0, delta.shape[0])
        expected = {
            name + ".weight": branch.delta().double()
            for name, branch in zip(installation.plan.target_names, installation.branches, strict=True)
        }
        native = convert_transformer(module, published_deltas, expected, external=True)
        assert native.keys() == expected.keys()
        for name, value in expected.items():
            torch.testing.assert_close(native[name], value, rtol=2e-6, atol=1e-7)
        assert identities == [id(p) for p in model.parameters()]
        for key, value in installation.branches.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "metadata,scale",
    [
        ({}, 1.0),
        ({"transformer.lora_alpha": 6}, 3.0),
        ({"transformer.lora_alpha": 6, "transformer.alpha_pattern": {"^block.to_q": 4}}, 2.0),
        ({"transformer.lora_alpha": 6, "transformer.use_rslora": True}, 6 / 2**0.5),
    ],
)
def test_peft_analysis_uses_official_scaling_metadata(tmp_path, metadata, scale):
    path = tmp_path / "peft.safetensors"
    save_tensors(
        {
            "transformer.block.to_q.lora_A.weight": torch.ones(2, 4),
            "transformer.block.to_q.lora_B.weight": torch.ones(6, 2),
        },
        path,
        metadata={"lora_adapter_metadata": json.dumps(metadata)},
    )
    with inspect_artifact(path) as artifact:
        name, parts = next(iter(groups(artifact).items()))
        delta = load_delta(artifact, name, parts)
        assert delta.scale == pytest.approx(scale)
        torch.testing.assert_close(delta.rows(0, 6), torch.full((6, 4), 2 * scale, dtype=torch.float64))
