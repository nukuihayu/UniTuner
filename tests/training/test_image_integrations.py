"""Exercise extracted model assembly and native/sliced adapter publication for image families."""

from contextlib import closing
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from tests.models.test_transformer_external_loading import _CASES
from unitunner.adapters.config import AdapterConfig
from unitunner.artifacts.tensors import TensorReader, iter_tensors
from unitunner.config.model import ModelConfig
from unitunner.models.catalog import ModelFamily
from unitunner.training.integrations.image_models import load_image_model
from unitunner.training.integrations.image_publication import publish_image_model
from unitunner.training.parameters import ParameterInstallation

IMAGE_CASES = [case for case in _CASES if case[0] not in ("wan", "minimax_h3")]


@pytest.mark.parametrize("module, stem, options", IMAGE_CASES)
@pytest.mark.parametrize("algorithm", [None, "lora", "loha", "lokr"])
def test_image_loading_and_publication_roundtrip(tmp_path, module, stem, options, algorithm):
    provider = import_module(f"unitunner.models.{module}.loading")
    topology = getattr(import_module(f"unitunner.models.{module}.config.model"), stem + "Config")(**options)
    original = getattr(import_module(f"unitunner.models.{module}.transformer"), stem + "Transformer")(topology)
    suffix = "qwen_image_21" if module == "qwen_image_2_1" else module
    source = tmp_path / "original.safetensors"
    getattr(provider, "save_" + suffix)(original, source)
    selection = ModelConfig(
        family=ModelFamily(module.replace("_", "-")),
        variant=getattr(topology, "variant", "base"),
        version=getattr(topology, "version", None),
        weights=str(source),
    )
    before_rng = torch.get_rng_state().clone()
    model, binding = load_image_model(selection, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert binding.model is model
    assert model.config == original.config
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name], atol=0, rtol=0)
    adapter = None if algorithm is None else AdapterConfig(algorithm=algorithm, rank=2, alpha=3.0)
    targets = {} if adapter is None else binding.adapter_targets()
    with closing(ParameterInstallation(model, adapter=adapter, targets=targets)) as installation:
        destination = tmp_path / "published.safetensors"
        metadata = {"custom": "retained"}
        publish_image_model(
            SimpleNamespace(train=SimpleNamespace(adapter=adapter), model=selection),
            model,
            binding,
            installation,
            destination,
            dtype=torch.bfloat16,
            metadata=metadata,
        )
        assert metadata == {"custom": "retained"}
        tensors = dict(iter_tensors(destination))
        assert tensors and all(value.dtype == torch.bfloat16 for value in tensors.values())
        with TensorReader(destination) as reader:
            info = reader.metadata()
        assert info["custom"] == "retained"
        if adapter is None:
            restored = getattr(provider, "load_" + suffix)(destination, dtype=torch.bfloat16)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[name].to(torch.bfloat16), atol=0, rtol=0)
        else:
            if module in ("flux", "flux2"):
                assert info["unitunner.adapter.layout"] == "flux-diffusers"
                assert not any("double_blocks" in key for key in tensors)
                assert any("attn_to_k" in key.replace(".", "_") for key in tensors)
            if algorithm == "lora":
                assert info["network_module"].startswith("networks.lora_")
            else:
                assert info["network_module"] == "lycoris.kohya"
