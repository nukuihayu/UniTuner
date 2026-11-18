"""Validate consumer projections, native reload and atomic attachment of fused FLUX adapters."""

from contextlib import closing

import pytest
import torch

from tests.support.assertions import require_type
from unituner.adapters.config import AdapterConfig
from unituner.adapters.loading import LoadedAdapter
from unituner.adapters.lokr import LoKr
from unituner.artifacts.formats.flux_adapters import flux_adapter_layout
from unituner.artifacts.formats.projection_slices import publish_sliced_adapter
from unituner.artifacts.tensors import save_tensors
from unituner.models.flux.config.model import FluxConfig
from unituner.models.flux.projections import adapter_projections
from unituner.models.flux.transformer import FluxTransformer
from unituner.training.parameters import ParameterInstallation


@pytest.mark.parametrize("algorithm,hidden_size", [("lora", 32), ("loha", 32), ("lokr", 48), ("lokr", 32)])
def test_fused_factors_publish_reload_and_input_gradients(tmp_path, algorithm, hidden_size):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(481)
        model = FluxTransformer(
            FluxConfig(
                in_channels=4,
                out_channels=4,
                hidden_size=hidden_size,
                num_heads=hidden_size // 16,
                axes_dim=(4, 6, 6),
                context_in_dim=6,
                vec_in_dim=8,
                depth=1,
                depth_single_blocks=1,
            )
        )
        targets = adapter_projections(model)
        config = AdapterConfig(algorithm=algorithm, rank=2, alpha=7.0)
        layout = flux_adapter_layout(targets, hidden_size=hidden_size, algorithm=algorithm)
        assert sum(map(len, layout.values())) == 14
        assert all(
            part.external.startswith("transformer." if algorithm == "lora" else "lycoris_")
            for parts in layout.values()
            for part in parts
        )
        state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        identities = tuple(id(value) for value in model.parameters())
        inputs = {name: torch.randn(2, 3, layer.in_features, requires_grad=True) for name, layer in targets.items()}
        with closing(
            ParameterInstallation(model, adapter=config, targets=targets, exclude_layers=("single_blocks",))
        ) as installation:
            selected_layout = {name: layout[name] for name in installation.plan.target_names}
            with torch.no_grad():
                for value in installation.branches.parameters():
                    value.normal_(0, 0.13)
            expected = {name: layer(inputs[name]) for name, layer in targets.items()}
            gradients = torch.autograd.grad(
                require_type(sum(value.square().mean() for value in expected.values()), torch.Tensor),
                tuple(inputs.values()),
            )
            published = publish_sliced_adapter(installation.branches.state_dict(), selected_layout, algorithm=algorithm)
            save_tensors(published, tmp_path / "adapter.safetensors")
            if algorithm == "lokr":
                native = installation.branches[0]
                assert isinstance(native, LoKr)
                q = layout[next(iter(targets))][0].external
                assert published[q + ".lokr_w2_a"].shape[1] == (
                    native.rank if hidden_size == 48 else native.rank * native.left.shape[1]
                )
        rng = torch.get_rng_state().clone()
        with closing(LoadedAdapter(tmp_path / "adapter.safetensors", config, targets, projection_layout=layout)):
            assert torch.equal(rng, torch.get_rng_state())
            actual = {name: layer(inputs[name]) for name, layer in targets.items()}
            observed_gradients = torch.autograd.grad(
                require_type(sum(value.square().mean() for value in actual.values()), torch.Tensor),
                tuple(inputs.values()),
            )
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=1e-5, rtol=1e-4)
            for actual_gradient, expected_gradient in zip(observed_gradients, gradients, strict=True):
                torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-5, rtol=1e-4)
        assert tuple(id(value) for value in model.parameters()) == identities
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, state[name], atol=0, rtol=0)
        # Bad slices must fail before attaching even the first valid projection.
        first = next(iter(layout.values()))
        published[first[1].external + ".alpha"] = torch.tensor(8.0)
        save_tensors(published, tmp_path / "bad.safetensors")
        with pytest.raises(ValueError, match="shared factor alpha"):
            LoadedAdapter(tmp_path / "bad.safetensors", config, targets, projection_layout=layout)
        assert all("forward" not in layer.__dict__ for layer in targets.values())
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        torch.set_num_threads(previous)
