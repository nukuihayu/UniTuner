"""Check H2D-only staging, replay, parameter identity and failure rollback on actual CUDA."""

from contextlib import ExitStack, closing
from copy import deepcopy

import pytest
import torch
from torch import nn

from tests.support.assertions import require_type
from unituner.adapters.config import AdapterConfig
from unituner.config.execution import ExecutionConfig
from unituner.execution.frozen_blocks import FrozenBlockInstallation
from unituner.execution.installation import validate_training_execution
from unituner.execution.regions import ModelRegion
from unituner.training.parameters import ParameterInstallation

pytestmark = pytest.mark.filterwarnings("error:Full backward hook is firing:UserWarning")


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr"])
def test_actual_h2d_backward_and_repeated_updates_match_resident(pinned, algorithm):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(81)
    resident = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Dropout(0.2), nn.Linear(8, 4)).to(device)
    staged = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Dropout(0.2), nn.Linear(8, 4)).to(device)
    staged.load_state_dict(resident.state_dict())
    with ExitStack() as owners:
        native = owners.enter_context(
            closing(
                ParameterInstallation(
                    resident,
                    adapter=AdapterConfig(algorithm=algorithm, rank=2),
                    targets={
                        "0": require_type(resident[0], torch.nn.Linear),
                        "3": require_type(resident[3], torch.nn.Linear),
                    },
                )
            )
        )
        streamed = owners.enter_context(
            closing(
                ParameterInstallation(
                    staged,
                    adapter=AdapterConfig(algorithm=algorithm, rank=2),
                    targets={
                        "0": require_type(staged[0], torch.nn.Linear),
                        "3": require_type(staged[3], torch.nn.Linear),
                    },
                )
            )
        )
        streamed.branches.load_state_dict(native.branches.state_dict())
        original = {key: value.clone() for key, value in staged.state_dict().items()}
        identities = [id(parameter) for parameter in staged.parameters()]
        owner = FrozenBlockInstallation(ModelRegion(staged, 0, "visual"), device=device, use_pinned_memory=pinned)
        owners.callback(owner.close)
        assert identities == [id(parameter) for parameter in staged.parameters()]
        assert all(
            parameter.device == device if parameter.requires_grad else parameter.device.type == "cpu"
            for parameter in staged.parameters()
        )
        assert all(parameter.is_pinned() == pinned for parameter in staged.parameters() if not parameter.requires_grad)
        optimizers = [torch.optim.AdamW(item.plan.parameters, lr=0.01) for item in (native, streamed)]
        for _ in range(3):
            inputs = torch.randn(2, 8, device=device)
            rng = torch.cuda.get_rng_state(device)
            expected = resident(inputs)
            expected.square().mean().backward()
            next_rng = torch.cuda.get_rng_state(device)
            torch.cuda.set_rng_state(rng, device)
            actual = staged(inputs)
            actual.square().mean().backward()
            assert torch.equal(next_rng, torch.cuda.get_rng_state(device))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for left, right in zip(native.plan.parameters, streamed.plan.parameters, strict=True):
                assert (left.grad is None) == (right.grad is None)
                if left.grad is not None:
                    torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        owner.close()
        owner.close()
        assert identities == [id(parameter) for parameter in staged.parameters()]
        assert all(parameter.device == device for parameter in staged.parameters())
        assert all(
            torch.equal(left, right)
            for left, right in zip(native.branches.parameters(), streamed.branches.parameters(), strict=True)
        )
        assert all(
            torch.equal(original[key], value)
            for key, value in staged.state_dict().items()
            if "parametrizations" not in key or key.endswith(".original")
        )


def test_invalid_policy_rejects_before_construction():
    with pytest.raises(ValueError, match="CUDA"):
        validate_training_execution(
            ExecutionConfig(precision="fp32", residency="frozen-blocks", recompute="block"),
            device=torch.device("cpu"),
            adapter=True,
        )
    with pytest.raises(ValueError, match="recompute"):
        validate_training_execution(
            ExecutionConfig(precision="fp32", residency="frozen-blocks", recompute="none"),
            device=torch.device("cuda"),
            adapter=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_no_input_gradient_replay_handles_partial_outputs_and_retained_graph():
    """Release replay leases when only one output participates in repeated backward."""

    class Branches(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8, device="cuda")
            self.linear.weight.requires_grad_(False)

        def forward(self, *, inputs):  # ty: ignore[invalid-mutable-override]
            hidden = self.linear(inputs)
            return hidden.square(), hidden.sin()

    model = Branches()
    reference = deepcopy(model)
    inputs = torch.randn(2, 8, device="cuda")
    with closing(FrozenBlockInstallation(ModelRegion(model, 0, "visual"), device=inputs.device)) as owner:
        loss = model(inputs=inputs)[1].sum()
        expected = reference(inputs=inputs)[1].sum()
        for retain in (True, False):
            loss.backward(retain_graph=retain)
            expected.backward(retain_graph=retain)
            torch.testing.assert_close(model.linear.bias.grad, reference.linear.bias.grad, atol=0, rtol=0)
            model.linear.bias.grad = reference.linear.bias.grad = None
            assert not owner._replays


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("cache_budget", [0, 32, 2048])
def test_mixed_dtype_strided_weights_and_outstanding_graphs(pinned, cache_budget, vbar=False):
    """Keep native strides, live trainable storage and independent backward lifetimes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda", torch.cuda.current_device())

    class MixedLinear(nn.Linear):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return (super().forward(input) + self.get_buffer("offset")) * require_type(self.scale, torch.Tensor)

    model = MixedLinear(8, 8, device=device)
    model.weight.requires_grad_(False)
    model.weight.data = model.weight.data.t()
    model.register_parameter(
        "scale", nn.Parameter(torch.randn(8, device=device, dtype=torch.float64), requires_grad=False)
    )
    model.register_buffer("offset", torch.randn(8, device=device))
    reference = deepcopy(model)
    identities = tuple(map(id, model.parameters()))
    stride = model.weight.stride()
    with closing(
        FrozenBlockInstallation(
            ModelRegion(model, 0, "visual"),
            device=device,
            use_pinned_memory=pinned,
            cache_budget_bytes=cache_budget,
            use_vbar=vbar,
        )
    ) as owner:
        if not vbar:
            assert owner._cache is not None
            assert owner._cache.statistics().allocated_bytes == 0
        assert model.weight.stride() == stride
        for _ in range(3):
            inputs = [torch.randn(2, 8, device=device, requires_grad=True) for _ in range(2)]
            copies = [value.detach().clone().requires_grad_() for value in inputs]
            outputs = [model(value) for value in inputs]
            expected = [reference(value) for value in copies]
            require_type(sum(value.square().mean() for value in outputs), torch.Tensor).backward()
            require_type(sum(value.square().mean() for value in expected), torch.Tensor).backward()
            for actual, wanted in zip(outputs, expected, strict=True):
                torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
            for actual, wanted in zip(inputs, copies, strict=True):
                torch.testing.assert_close(actual.grad, wanted.grad, atol=0, rtol=0)
            torch.testing.assert_close(model.bias.grad, reference.bias.grad, atol=0, rtol=0)
            with torch.no_grad():
                assert model.bias.grad is not None
                model.bias.add_(model.bias.grad, alpha=-0.01)
                assert reference.bias.grad is not None
                reference.bias.add_(reference.bias.grad, alpha=-0.01)
            model.bias.grad = reference.bias.grad = None
    assert tuple(map(id, model.parameters())) == identities
    assert model.weight.stride() == stride
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)


def test_offload_defaults_require_no_cache_or_recompute_tuning():
    for residency in ("frozen-blocks", "vbar"):
        config = ExecutionConfig(residency=residency)
        assert config.recompute == "block"
        assert config.recompute_interval == 1
        assert config.frozen_weight_cache_mb is None
        assert config.use_pinned_memory_for_block_swap is None
    assert ExecutionConfig().recompute == "none"
