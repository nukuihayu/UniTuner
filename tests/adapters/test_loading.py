"""Exercise published adapter reuse in native inference without a trainer runtime."""

import pytest
import torch
from torch import nn

from unituner.adapters.config import AdapterConfig
from unituner.adapters.loading import LoadedAdapter
from unituner.artifacts.formats.adapters import AdapterTargetKeys, adapter_key_map, convert_adapter_keys
from unituner.artifacts.tensors import save_tensors
from unituner.training.parameters import ParameterInstallation


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr"])
def test_export_load_preserves_nonzero_prediction_input_gradient_and_base(tmp_path, algorithm):
    torch.manual_seed(52)
    model = nn.Sequential(nn.Linear(6, 8, dtype=torch.float64), nn.SiLU(), nn.Linear(8, 3, dtype=torch.float64))
    first, last = model[0], model[2]
    assert isinstance(first, nn.Linear) and isinstance(last, nn.Linear)
    targets = {"first": first, "last": last}
    prefixes = {name: "lora_unet_" + name for name in targets}
    config = AdapterConfig(algorithm=algorithm, rank=2, alpha=7.0)
    inputs = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    base = model(inputs)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    installation = ParameterInstallation(
        model, adapter=config, targets=targets, include_layers=("*",), exclude_layers=("first",)
    )
    assert installation.plan.target_names == ("last",)
    assert "forward" not in model[0].__dict__
    with torch.no_grad():
        for parameter in installation.branches.parameters():
            parameter.normal_(std=0.25)
    expected = model(inputs)
    expected_gradient = torch.autograd.grad(expected.sum(), inputs)[0]
    assert not torch.equal(expected, base)
    mapping = adapter_key_map(
        [AdapterTargetKeys(str(index), prefixes[name]) for index, name in enumerate(installation.plan.target_names)],
        algorithm=algorithm,
    )
    tensors = convert_adapter_keys(installation.branches.state_dict(), mapping)
    path = tmp_path / "adapter.safetensors"
    save_tensors(tensors, path)
    installation.close()
    rng = torch.get_rng_state().clone()
    native_ids = [id(parameter) for parameter in model.parameters()]
    loaded = LoadedAdapter(path, config.model_copy(update={"alpha": 1.0}), targets, external_prefixes=prefixes)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(loaded.branches) == 1
    actual = model(inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), inputs)[0], expected_gradient, atol=0, rtol=0)
    loaded.close()
    loaded.close()
    torch.testing.assert_close(model(inputs), base, atol=0, rtol=0)
    assert [id(parameter) for parameter in model.parameters()] == native_ids
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)
    tensors[next(key for key in tensors if not key.endswith(".alpha"))] = torch.zeros(1)
    save_tensors(tensors, path)
    with pytest.raises(ValueError, match="mismatch"):
        LoadedAdapter(path, config, targets, external_prefixes=prefixes)
    torch.testing.assert_close(model(inputs), base, atol=0, rtol=0)
