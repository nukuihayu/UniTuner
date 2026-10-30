"""Check selected updates, frozen exclusions and transactional selection failures."""

from contextlib import closing

import pytest
import torch
from torch import nn

from tests.support.assertions import require_type
from unituner.adapters.config import AdapterConfig
from unituner.training.parameters import ParameterInstallation


def test_full_selection_updates_only_selected_native_parameters():
    torch.manual_seed(236)
    model = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 2))
    model[0].weight.requires_grad_(False)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    flags = tuple(parameter.requires_grad for parameter in model.parameters())
    identities = tuple(id(parameter) for parameter in model.parameters())
    with closing(ParameterInstallation(model, include_layers=("2",), exclude_layers=("2.bias",))) as owner:
        assert owner.plan.target_names == ("2.weight",)
        optimizer = torch.optim.SGD(owner.plan.parameters, lr=0.1)
        model(torch.randn(3, 4)).square().sum().backward()
        assert model[0].weight.grad is None and model[0].bias.grad is None and model[2].bias.grad is None
        optimizer.step()
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter, before[name]) == (name != "2.weight")
    assert tuple(parameter.requires_grad for parameter in model.parameters()) == flags
    assert tuple(id(parameter) for parameter in model.parameters()) == identities


@pytest.mark.parametrize("include,exclude", [(("missing.*",), ()), ((), ("*",)), (("0",), ("missing",))])
def test_invalid_selection_preserves_rng_trainability_and_forwards(include, exclude):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    flags = tuple(parameter.requires_grad for parameter in model.parameters())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Layer"):
        ParameterInstallation(
            model,
            adapter=AdapterConfig(rank=2, alpha=7.0),
            targets={"0": require_type(model[0], torch.nn.Linear), "1": require_type(model[1], torch.nn.Linear)},
            include_layers=include,
            exclude_layers=exclude,
        )
    assert torch.equal(torch.get_rng_state(), rng)
    assert tuple(parameter.requires_grad for parameter in model.parameters()) == flags
    assert all("forward" not in layer.__dict__ for layer in model)
