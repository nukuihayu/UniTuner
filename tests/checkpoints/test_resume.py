"""Compare actual stochastic continuation against saved-state restoration."""

import random
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch import nn

from unituner.adapters.attachment import AdapterAttachment
from unituner.adapters.lora import LoRA
from unituner.checkpoints.restore import restore_checkpoint
from unituner.checkpoints.saving import save_checkpoint
from unituner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unituner.training.contracts import RunState, StepResult
from unituner.training.engine import update_phase
from unituner.training.losses.flow_matching import flow_matching_loss
from unituner.training.session import TrainingSession

IDENTITY = ResumeIdentity(config_sha256="a" * 64, dataset_manifest_sha256="b" * 64, model_artifact_sha256="c" * 64)


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("named_optimizer", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_compact_resume_reproduces_updates_and_allocated_adam_state(tmp_path, frozen, named_optimizer, dtype):
    """Compare exact continuation with populated Adam state and all host RNGs."""
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    model = nn.Sequential(nn.Linear(4, 4, dtype=dtype), nn.Dropout(0.25)).requires_grad_(not frozen)
    adapter = nn.Linear(4, 4, dtype=dtype)
    modules = {"model": model, "adapter": adapter}
    # A new session must load exactly the same frozen base, not a random one.
    fresh_modules = deepcopy(modules)

    def make_optimizer(items):
        parameters = [p for module in items.values() for p in module.parameters() if p.requires_grad]
        if named_optimizer:
            return {
                "first": torch.optim.AdamW(parameters[:1], lr=0.003),
                "second": torch.optim.AdamW(parameters[1:], lr=0.002),
            }
        return torch.optim.AdamW(parameters, lr=0.003)

    def advance_direct(items, optimizers):
        inputs = torch.from_numpy(np.random.standard_normal((3, 4))).to(dtype) * random.random()
        target = torch.randn(3, 4, dtype=dtype)
        prediction = items["adapter"](items["model"](inputs))
        (prediction.float() - target.float()).square().mean().backward()
        roles = optimizers.values() if isinstance(optimizers, dict) else [optimizers]
        for optimizer in roles:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    optimizer = make_optimizer(modules)
    fresh_optimizer = make_optimizer(fresh_modules)
    advance_direct(modules, optimizer)
    path = tmp_path / "compact.safetensors"
    cursor = CheckpointCursor(updates=1, microbatches=1, samples=3)
    save_checkpoint(path, modules, optimizer, cursor, IDENTITY, reference_frozen_parameters=True)
    advance_direct(modules, optimizer)
    advance_direct(modules, optimizer)
    expected_rng = (random.random(), np.random.random(), torch.rand(2))
    assert restore_checkpoint(path, fresh_modules, fresh_optimizer, IDENTITY) == cursor
    advance_direct(fresh_modules, fresh_optimizer)
    advance_direct(fresh_modules, fresh_optimizer)
    actual_rng = (random.random(), np.random.random(), torch.rand(2))
    assert expected_rng[:2] == actual_rng[:2]
    torch.testing.assert_close(expected_rng[2], actual_rng[2], atol=0, rtol=0)
    for name, module in modules.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, fresh_modules[name].state_dict()[key], atol=0, rtol=0)
    expected_roles = optimizer if isinstance(optimizer, dict) else {"main": optimizer}
    actual_roles = fresh_optimizer if isinstance(fresh_optimizer, dict) else {"main": fresh_optimizer}
    for name, item in expected_roles.items():
        expected, actual = item.state_dict(), actual_roles[name].state_dict()
        assert expected["param_groups"] == actual["param_groups"]
        assert expected["state"].keys() == actual["state"].keys()
        for index, values in expected["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(value, actual["state"][index][key], atol=0, rtol=0)


def build(session, adapter):
    model = nn.Sequential(nn.Linear(4, 4, dtype=torch.float64), nn.Dropout(0.25))
    modules: dict[str, nn.Module] = {"model": model}
    if adapter:
        model.requires_grad_(False)
        branch = LoRA(4, 4, rank=2, alpha=5.0, dtype=torch.float64)
        with torch.no_grad():
            branch.up.normal_(0, 0.2)
        projection = model[0]
        assert isinstance(projection, nn.Linear)
        session.own(AdapterAttachment(projection, branch))
        modules["adapter"] = branch
        parameters = list(branch.parameters())
    else:
        parameters = list(model.parameters())
    return model, modules, torch.optim.AdamW(parameters, lr=0.003)


def advance(model, optimizer, state):
    # Every supported host RNG affects the next mathematical update.
    inputs = torch.from_numpy(np.random.standard_normal((3, 4))) * random.random()
    target = torch.randn(3, 4, dtype=torch.float64)
    update_phase([StepResult(flow_matching_loss(model(inputs), target), 3)], optimizer, state)


@pytest.mark.parametrize("adapter", [False, True])
def test_stochastic_resume_matches_continuous_updates(tmp_path, adapter):
    random.seed(22)
    np.random.seed(22)
    torch.manual_seed(22)
    with TrainingSession() as session, TrainingSession() as restored_session:
        model, modules, optimizer = build(session, adapter)
        state = RunState()
        advance(model, optimizer, state)
        path = tmp_path / "state.safetensors"
        save_checkpoint(path, modules, optimizer, CheckpointCursor(**asdict(state)), IDENTITY)
        checkpoint_bytes = path.read_bytes()
        advance(model, optimizer, state)
        advance(model, optimizer, state)
        expected_rng = (random.random(), np.random.random(), torch.rand(2))
        fresh, fresh_modules, fresh_optimizer = build(restored_session, adapter)
        parameter_ids = [id(parameter) for group in fresh_optimizer.param_groups for parameter in group["params"]]
        cursor = restore_checkpoint(path, fresh_modules, fresh_optimizer, IDENTITY)
        fresh_state = RunState(**cursor.model_dump())
        advance(fresh, fresh_optimizer, fresh_state)
        advance(fresh, fresh_optimizer, fresh_state)
        actual_rng = (random.random(), np.random.random(), torch.rand(2))
        assert fresh_state == state
        assert actual_rng[:2] == expected_rng[:2]
        torch.testing.assert_close(actual_rng[2], expected_rng[2], atol=0, rtol=0)
        assert parameter_ids == [
            id(parameter) for group in fresh_optimizer.param_groups for parameter in group["params"]
        ]
        for name, module in modules.items():
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, fresh_modules[name].state_dict()[key], atol=0, rtol=0)
        expected_optimizer, actual_optimizer = optimizer.state_dict(), fresh_optimizer.state_dict()
        assert expected_optimizer["param_groups"] == actual_optimizer["param_groups"]
        for index, values in expected_optimizer["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(value, actual_optimizer["state"][index][key], atol=0, rtol=0)
        assert path.read_bytes() == checkpoint_bytes


def test_wrong_numerical_identity_rejected_before_mutation(tmp_path):
    with TrainingSession() as session:
        model, modules, optimizer = build(session, False)
        path = tmp_path / "state.safetensors"
        save_checkpoint(path, modules, optimizer, CheckpointCursor(updates=0, microbatches=0, samples=0), IDENTITY)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        wrong = IDENTITY.model_copy(update={"config_sha256": "d" * 64})
        rng = torch.get_rng_state().clone()
        with pytest.raises(ValueError, match="numerical identity"):
            restore_checkpoint(path, modules, optimizer, wrong)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


def test_changed_optimizer_roles_rejected_before_mutation(tmp_path):
    with TrainingSession() as session:
        model, modules, optimizer = build(session, False)
        path = tmp_path / "state.safetensors"
        save_checkpoint(path, modules, optimizer, CheckpointCursor(updates=0, microbatches=0, samples=0), IDENTITY)
        reversed_optimizer = torch.optim.AdamW(list(reversed(list(model.parameters()))), lr=0.003)
        with pytest.raises(ValueError, match="parameter roles"):
            restore_checkpoint(path, modules, reversed_optimizer, IDENTITY)


def test_frozen_references_reduce_payload_validate_base_and_restore_mutable_buffers(tmp_path):
    """Omit frozen parameter bytes only; retain evolving buffers and reject changed bases before mutation."""
    base = nn.Linear(1024, 1024).requires_grad_(False)
    base.register_buffer("running", torch.tensor([3.0]))
    adapter = nn.Linear(2, 2)
    teacher = nn.Linear(2, 2).requires_grad_(False)
    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
    modules = {"model": base, "adapter": adapter, "teacher": teacher}
    optimizer = torch.optim.AdamW(adapter.parameters())
    cursor = CheckpointCursor(updates=0, microbatches=0, samples=0)
    complete, compact = tmp_path / "complete.safetensors", tmp_path / "compact.safetensors"
    save_checkpoint(complete, modules, optimizer, cursor, IDENTITY)
    save_checkpoint(compact, modules, optimizer, cursor, IDENTITY, reference_frozen_parameters=True)
    assert compact.stat().st_size < complete.stat().st_size // 10
    before = {key: value.clone() for key, value in adapter.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        base.weight[0, 0].add_(1)
    with pytest.raises(ValueError, match="base content"):
        restore_checkpoint(compact, modules, optimizer, IDENTITY)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, adapter.state_dict()[key]) for key, value in before.items())
    # Reload the exact reference, then change only mutable state.
    restore_checkpoint(complete, modules, optimizer, IDENTITY)
    with torch.no_grad():
        base.get_buffer("running").fill_(9)
        teacher.weight.fill_(8)
        adapter.weight.fill_(5)
    identities = [id(parameter) for module in modules.values() for parameter in module.parameters()]
    restore_checkpoint(compact, modules, optimizer, IDENTITY)
    assert base.get_buffer("running").item() == 3
    assert all(torch.equal(value, teacher.state_dict()[key]) for key, value in teacher_before.items())
    assert identities == [id(parameter) for module in modules.values() for parameter in module.parameters()]
    assert all(torch.equal(value, adapter.state_dict()[key]) for key, value in before.items())
