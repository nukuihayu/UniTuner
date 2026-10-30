"""Round-trip native published adapters and restore projections after inference failures."""

from contextlib import closing
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from tests.training.test_image_integrations import IMAGE_CASES
from unitunner.adapters.config import AdapterConfig
from unitunner.config.generation import GenerationAdapterConfig
from unitunner.config.output import OutputConfig
from unitunner.inference.adapters import image_adapter
from unitunner.training.integrations.image_models import bind_image_model
from unitunner.training.integrations.image_publication import publish_image_model
from unitunner.training.parameters import ParameterInstallation


@pytest.mark.parametrize("module,stem,options", IMAGE_CASES)
@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr"])
def test_native_image_adapter_roundtrip_and_failure_cleanup(tmp_path, module, stem, options, algorithm):
    topology = getattr(import_module(f"unitunner.models.{module}.config.model"), stem + "Config")(**options)
    model = getattr(import_module(f"unitunner.models.{module}.transformer"), stem + "Transformer")(topology).eval()
    binding = bind_image_model(model)
    targets = binding.adapter_targets()
    name, projection = next(iter(targets.items()))
    adapter = AdapterConfig(algorithm=algorithm, rank=2, alpha=3.0)
    inputs = torch.randn(2, projection.in_features)
    base = projection(inputs).detach()
    path = tmp_path / "adapter.safetensors"
    with closing(ParameterInstallation(model, adapter=adapter, targets={name: projection})) as installation:
        with torch.no_grad():
            for parameter in installation.branches.parameters():
                parameter.fill_(0.2)
        expected = projection(inputs).detach()
        assert not torch.equal(expected, base)
        publish_image_model(
            SimpleNamespace(
                train=SimpleNamespace(adapter=adapter),
                model=SimpleNamespace(family=module.replace("_", "-")),
                output=OutputConfig(),
            ),
            model,
            binding,
            installation,
            path,
            dtype=torch.float32,
            metadata={},
        )
    original_forward = projection.forward
    identities = tuple(id(parameter) for parameter in model.parameters())
    flags = tuple(parameter.requires_grad for parameter in model.parameters())
    rng = torch.get_rng_state().clone()
    config = GenerationAdapterConfig(weights=str(path), config=adapter)
    with pytest.raises(RuntimeError, match="sampling failed"), image_adapter(model, config):
        torch.testing.assert_close(projection(inputs), expected, atol=1e-5, rtol=1e-5)
        raise RuntimeError("sampling failed")
    assert projection.forward == original_forward
    torch.testing.assert_close(projection(inputs), base, atol=0, rtol=0)
    assert tuple(id(parameter) for parameter in model.parameters()) == identities
    assert tuple(parameter.requires_grad for parameter in model.parameters()) == flags
    assert torch.equal(torch.get_rng_state(), rng)
    with image_adapter(model, config):
        torch.testing.assert_close(projection(inputs), expected, atol=1e-5, rtol=1e-5)
    assert projection.forward == original_forward
