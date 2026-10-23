"""Verify FLUX.2 strict weight ownership, published geometry and adapter gradients."""

from contextlib import closing

import pytest
import torch

from tests.support.assertions import require_value
from unitunner.adapters.config import AdapterConfig
from unitunner.artifacts.tensors import save_tensors
from unitunner.models.flux2.config.model import Flux2Config, flux2_config
from unitunner.models.flux2.loading import load_flux2, save_flux2
from unitunner.models.flux2.projections import adapter_projections
from unitunner.models.flux2.transformer import Flux2Transformer
from unitunner.training.parameters import ParameterInstallation


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "version,variant,algorithm",
    [("dev", "distilled", "lora"), ("klein-4b", "base", "loha"), ("klein-9b", "distilled", "lokr")],
)
def test_flux2_strict_roundtrip_and_nonzero_adapter_gradients(tmp_path, version, variant, algorithm):
    config = Flux2Config(
        version=version,
        variant=variant,
        in_channels=8,
        context_in_dim=12,
        hidden_size=32,
        num_heads=2,
        depth=1,
        depth_single_blocks=1,
        axes_dim=(4, 4, 4, 4),
    )
    model = Flux2Transformer(config)
    path = tmp_path / "model.safetensors"
    rng = torch.get_rng_state().clone()
    save_flux2(model, path)
    restored = load_flux2(path)
    assert torch.equal(rng, torch.get_rng_state())
    assert restored.config == config
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    values = (
        torch.randn(2, 6, 8),
        torch.randn(2, 6, 4),
        torch.tensor([0.244, 0.751]),
        torch.randn(2, 5, 12),
        torch.randn(2, 5, 4),
        torch.tensor([2.5, 3.5]) if version == "dev" else None,
    )
    expected = model(*values)
    assert torch.equal(expected, restored(*values))
    targets = adapter_projections(restored)
    assert len(targets) == 10
    assert all("modulation" not in name for name in targets)
    identities = tuple(map(id, restored.parameters()))
    with closing(
        ParameterInstallation(restored, adapter=AdapterConfig(algorithm=algorithm, rank=2, alpha=2.0), targets=targets)
    ) as owner:
        with torch.no_grad():
            for parameter in owner.branches.parameters():
                parameter.normal_(0, 0.03)
        actual = restored(*values)
        assert not torch.equal(expected, actual)
        actual.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in owner.plan.parameters)
        assert any(torch.count_nonzero(require_value(p.grad)) for p in owner.plan.parameters)
    assert identities == tuple(map(id, restored.parameters()))
    assert torch.equal(expected, restored(*values))
    tensors = model.state_dict()
    tensors["img_in.weight"] = torch.full_like(tensors["img_in.weight"], float("nan"))
    save_tensors(tensors, path)
    with pytest.raises(ValueError, match="invalid"):
        load_flux2(path, config=config)
    tensors = model.state_dict()
    tensors.pop("img_in.weight")
    save_tensors(tensors, path)
    with pytest.raises(ValueError, match="keys differ"):
        load_flux2(path, config=config)
    save_flux2(model, path)
    with pytest.raises(ValueError, match="differs from artifact"):
        load_flux2(path, config=config.model_copy(update={"context_in_dim": 14}))
    with pytest.raises(ValueError, match="guidance"):
        restored(*values[:-1], None if version == "dev" else torch.ones(2))


def test_flux2_published_architecture_and_variant_contracts():
    for version, context, width, heads, double, single in (
        ("dev", 15360, 6144, 48, 8, 48),
        ("klein-4b", 7680, 3072, 24, 5, 20),
        ("klein-9b", 12288, 4096, 32, 8, 24),
    ):
        assert version in ("dev", "klein-4b", "klein-9b")
        config = flux2_config(version)
        assert (
            config.context_in_dim,
            config.hidden_size,
            config.num_heads,
            config.depth,
            config.depth_single_blocks,
        ) == (context, width, heads, double, single)
        with torch.device("meta"):
            model = Flux2Transformer(config)
        state = model.state_dict()
        assert state["img_in.weight"].shape == (width, 128)
        assert state["single_blocks.0.linear1.weight"].shape == (9 * width, width)
        assert not any(key.endswith("bias") for key in state)
        assert ("guidance_in.in_layer.weight" in state) == (version == "dev")
        if version != "dev":
            base = flux2_config(version, variant="base")
            with torch.device("meta"):
                base_state = Flux2Transformer(base).state_dict()
            assert {k: v.shape for k, v in state.items()} == {k: v.shape for k, v in base_state.items()}
    with pytest.raises(ValueError, match="guidance-distilled"):
        flux2_config("dev", variant="base")
    with pytest.raises(ValueError, match="rotary"):
        Flux2Config(axes_dim=(31, 33, 32, 32))
