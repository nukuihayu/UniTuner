"""Check actual gradients, forward ownership and cleanup after setup failures."""

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from unitunner.adapters.attachment import AdapterAttachment
from unitunner.adapters.lora import LoRA
from unitunner.config.execution import ExecutionConfig
from unitunner.execution.installation import ExecutionInstallation
from unitunner.execution.recompute import RecomputeInstallation, run_region
from unitunner.execution.regions import ModelRegion, ModelRegions
from unitunner.training.session import TrainingSession


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_default_execution_keeps_native_callable_and_gradients(monkeypatch, device):
    """Default policy must not invoke Dynamo even when the selected device is CUDA."""

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("Default execution must remain eager")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    model = nn.Linear(4, 3)
    regions = ModelRegions((ModelRegion(model, 0, "visual"),))
    original = model.forward
    installation = ExecutionInstallation(regions, ExecutionConfig(), device=torch.device(device))
    try:
        assert model.forward == original
        values = torch.randn(2, 4, requires_grad=True)
        model(values).square().sum().backward()
        assert values.grad is not None
        assert model.weight.grad is not None
    finally:
        installation.close()
    assert model.forward == original


@pytest.mark.parametrize("instance_forward", [False, True])
def test_attachment_keys_identity_and_reverse_ownership(instance_forward):
    torch.manual_seed(7)
    model = nn.Linear(4, 4)
    if instance_forward:
        native = model.forward
        model.forward = lambda values: native(values) * 2
    initial_forward = model.forward
    keys = tuple(model.state_dict())
    parameter_ids = tuple(map(id, model.parameters()))
    inputs = torch.randn(2, 4)
    original = model(inputs).detach()
    branch = LoRA(4, 4, rank=2, alpha=2.0)
    with torch.no_grad():
        branch.up.fill_(0.2)
    first = AdapterAttachment(model, branch)
    second = AdapterAttachment(model, branch)
    with pytest.raises(RuntimeError, match="later forward owners"):
        first.close()
    second.close()
    second.close()
    first.close()
    first.close()
    assert model.forward == initial_forward
    assert ("forward" in model.__dict__) == instance_forward
    assert tuple(model.state_dict()) == keys
    assert tuple(map(id, model.parameters())) == parameter_ids
    torch.testing.assert_close(model(inputs), original, atol=0, rtol=0)


def test_setup_failure_restores_forward():
    model = nn.Linear(3, 2)
    assert "forward" not in model.__dict__
    session = TrainingSession()
    with pytest.raises(RuntimeError, match="setup failure"):
        with session:
            session.own(AdapterAttachment(model, LoRA(3, 2, rank=1, alpha=1.0)))
            raise RuntimeError("setup failure")
    assert "forward" not in model.__dict__
    session.close()


def test_cleanup_failure_does_not_leave_other_adapter_installed():
    class FailingResource:
        def close(self):
            raise RuntimeError("close failure")

    model = nn.Linear(3, 2)
    session = TrainingSession()
    session.own(AdapterAttachment(model, LoRA(3, 2, rank=1, alpha=1.0)))
    session.own(FailingResource())
    with pytest.raises(RuntimeError, match="close failure"):
        session.close()
    assert "forward" not in model.__dict__
    session.close()


def test_recompute_dropout_and_reused_module_regions():
    torch.manual_seed(43)
    model = nn.Sequential(nn.Linear(4, 4, dtype=torch.float64), nn.Dropout(0.25), nn.SiLU())
    reference = copy.deepcopy(model)
    inputs = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    ref_input = inputs.detach().clone().requires_grad_()
    rng = torch.get_rng_state()
    # Reuse the same module with distinct branch identities and logical indices.
    first = ModelRegion(model, 0, "condition")
    second = ModelRegion(model, 1, "visual")
    actual = run_region(second, run_region(first, inputs, recompute_interval=2), recompute_interval=2)
    actual.square().sum().backward()
    torch.set_rng_state(rng)
    expected = reference(reference(ref_input))
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    for value, ref in zip((inputs, *model.parameters()), (ref_input, *reference.parameters()), strict=True):
        torch.testing.assert_close(value.grad, ref.grad, atol=1e-10, rtol=1e-8)


def test_compile_recompute_and_adapter_combination():
    from unitunner.execution.compiler import CompilerInstallation

    torch.manual_seed(9)
    model = nn.Linear(4, 4, dtype=torch.float64)
    model.requires_grad_(False)
    branch = LoRA(4, 4, rank=2, alpha=2.0, dtype=torch.float64)
    with torch.no_grad():
        branch.up.normal_(0, 0.2)
    reference = copy.deepcopy(model)
    ref_branch = copy.deepcopy(branch)
    inputs = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    ref_input = inputs.detach().clone().requires_grad_()
    identities = tuple(map(id, model.parameters()))
    keys = tuple(model.state_dict())
    with TrainingSession() as session:
        session.own(AdapterAttachment(model, branch))
        session.own(RecomputeInstallation(ModelRegion(model, 0, "visual")))
        session.own(CompilerInstallation(model, backend="eager"))
        session.own(AdapterAttachment(reference, ref_branch))
        actual = model(inputs)
        expected = reference(ref_input)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for value, ref in zip((inputs, *branch.parameters()), (ref_input, *ref_branch.parameters()), strict=True):
            torch.testing.assert_close(value.grad, ref.grad, atol=1e-10, rtol=1e-8)
    assert tuple(map(id, model.parameters())) == identities
    assert tuple(model.state_dict()) == keys
    assert "forward" not in model.__dict__


def test_preview_exception_restores_rng_and_mixed_child_flags():
    from unitunner.training.previews.context import preview_context

    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    model[0].eval()
    flags = [child.training for child in model.modules()]
    branches = nn.Sequential(nn.Dropout(0.5), nn.Linear(2, 2))
    branches[1].eval()
    branch_flags = [child.training for child in branches.modules()]
    rng = torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError, match="preview failure"):
        with preview_context(model, additional_models=(branches,)):
            assert not model.training
            assert not branches.training
            torch.rand(20)
            random.random()
            np.random.random(20)
            raise RuntimeError("preview failure")
    assert [child.training for child in model.modules()] == flags
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert [child.training for child in branches.modules()] == branch_flags
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0] and after_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])


def test_compiled_recompute_owners_isolate_heterogeneous_region_guard_budgets():
    from unitunner.execution.compiler import CompilerInstallation

    torch.manual_seed(91)
    models = [nn.Sequential(nn.Linear(size, size), nn.SiLU()) for size in range(3, 15)]
    references = copy.deepcopy(models)
    originals = [model.forward for model in models]
    with TrainingSession() as session:
        for index, (model, reference) in enumerate(zip(models, references, strict=True)):
            session.own(RecomputeInstallation(ModelRegion(model, index, "visual")))
            session.own(CompilerInstallation(model, backend="eager", fullgraph=True))
            inputs = torch.randn(2, index + 3, requires_grad=True)
            ref_inputs = inputs.detach().clone().requires_grad_()
            actual, expected = model(inputs), reference(ref_inputs)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for value, ref in zip((inputs, *model.parameters()), (ref_inputs, *reference.parameters()), strict=True):
                torch.testing.assert_close(value.grad, ref.grad, atol=0, rtol=0)
    assert all(
        model.forward == original and "forward" not in model.__dict__
        for model, original in zip(models, originals, strict=True)
    )


def test_compiler_scoped_settings_restore_after_failed_training_and_evaluation():
    from torch._dynamo import config as dynamo_config
    from torch._functorch import config as autograd_config

    from unitunner.execution.compiler import CompilerInstallation

    class Failing(torch.nn.Module):
        def forward(self, values):
            if self.training:
                assert autograd_config.donated_buffer == previous[0]
                assert autograd_config.backward_pass_autocast == "off"
                assert dynamo_config.fail_on_recompile_limit_hit
                raise ValueError("training failure")
            return values + 1

    model = Failing()
    previous = (
        autograd_config.donated_buffer,
        dynamo_config.recompile_limit,
        dynamo_config.fail_on_recompile_limit_hit,
        autograd_config.backward_pass_autocast,
    )
    owner = CompilerInstallation(model, backend="eager")
    try:
        with pytest.raises(ValueError, match="training failure"):
            model(torch.ones(2))
        model.eval()
        torch.testing.assert_close(model(torch.ones(2)), torch.full((2,), 2.0))
        assert previous == (
            autograd_config.donated_buffer,
            dynamo_config.recompile_limit,
            dynamo_config.fail_on_recompile_limit_hit,
            autograd_config.backward_pass_autocast,
        )
    finally:
        owner.close()
