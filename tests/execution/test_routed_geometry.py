"""Static routing metadata is shared across blocks, while activations stay differentiable."""

from collections import Counter

import pytest
import torch

from tests.execution.test_native_strategies import _case
from tests.models.test_minimax_h3_image_training import _binding
from unituner.config.routing import TreadConfig
from unituner.execution.routing import TokenRoute
from unituner.training.session import TrainingSession


def test_minimax_image_builds_segments_on_cpu(monkeypatch):
    from unituner.models.minimax_h3 import image

    binding = _binding()
    original = image._build_modulation_segments
    observed = []

    def observe(indices):
        observed.append(indices.device.type)
        return original(indices)

    monkeypatch.setattr(image, "_build_modulation_segments", observe)
    latents = torch.randn(1, 24, 1, 2, 4, requires_grad=True)
    output = binding.model.predict(
        latents,
        torch.randn(1, 3, 8),
        torch.tensor([[1, 0, 1]]),
        sigma=0.75,
        sigma_next=0.5,
        references=(torch.randn_like(latents),),
    )
    output.square().mean().backward()
    assert observed == ["cpu"]
    assert latents.grad is not None and torch.isfinite(latents.grad).all()


@pytest.mark.parametrize("family", ["flux", "flux2", "qwen-image", "krea2", "z-image", "moss-image", "minimax-h3"])
def test_static_geometry_selected_once_per_forward(family, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(941)
    selected = Counter()
    geometry = []
    original = TokenRoute.select

    def observe(self, value, **kwargs):
        if not value.requires_grad:
            geometry.append(value)
            selected[id(value)] += 1
        return original(self, value, **kwargs)

    monkeypatch.setattr(TokenRoute, "select", observe)
    with TrainingSession() as session:
        binding, modules, request = _case(family, session)
        binding.model.train()
        binding.model.token_routing = TreadConfig(end_layer=2)
        for _ in range(2):
            selected.clear()
            geometry.clear()
            loss = binding.step(request).loss
            loss.backward()
            assert torch.isfinite(loss)
            assert selected and max(selected.values()) == 1
            assert any(parameter.grad is not None for parameter in modules["adapter"].parameters())
            for module in modules.values():
                module.zero_grad(set_to_none=True)
