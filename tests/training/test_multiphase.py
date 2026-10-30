"""Verify interleaved GAN accumulation and exact native multi-optimizer resume."""

import copy
from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from unitunner.checkpoints.restore import restore_checkpoint
from unitunner.checkpoints.saving import save_checkpoint
from unitunner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unitunner.training.contracts import RunState, StepResult
from unitunner.training.engine import update_phases
from unitunner.training.losses.adversarial import frozen_discriminator, gan_loss

IDENTITY = ResumeIdentity(config_sha256="a" * 64, dataset_manifest_sha256="b" * 64, model_artifact_sha256="c" * 64)


def build():
    generator = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.2), nn.Linear(4, 3))
    discriminator = spectral_norm(nn.Linear(3, 2))
    modules = {"generator": generator, "discriminator": discriminator}
    optimizers = {name: torch.optim.AdamW(module.parameters(), lr=0.003) for name, module in modules.items()}
    return modules, optimizers


def steps(modules):
    for count in (2, 3):
        low, high = torch.randn(count, 3), torch.randn(count, 3)
        generated = modules["generator"](low)
        with frozen_discriminator(modules["discriminator"]):
            loss = F.mse_loss(generated, high) + 0.3 * gan_loss((modules["discriminator"](generated),), True)
        yield StepResult(loss, count, "generator")
        fake = modules["discriminator"](generated.detach())
        real = modules["discriminator"](high)
        yield StepResult(0.3 * (gan_loss((fake,), False) + gan_loss((real,), True)), count, "discriminator")


def test_three_gan_updates_match_independent_interleaved_accumulation():
    torch.manual_seed(129)
    modules, optimizers = build()
    reference = copy.deepcopy(modules)
    reference_optimizers = {
        name: torch.optim.AdamW(module.parameters(), lr=0.003) for name, module in reference.items()
    }
    state = RunState()
    for _ in range(3):
        rng = torch.get_rng_state()
        result = update_phases(steps(modules), optimizers, state)
        assert set(result) == {"generator", "discriminator"}
        torch.set_rng_state(rng)
        for optimizer in reference_optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        for item in steps(reference):
            (item.loss * item.sample_count / 5).backward()
        for optimizer in reference_optimizers.values():
            optimizer.step()
        for name in modules:
            for key, tensor in modules[name].state_dict().items():
                torch.testing.assert_close(tensor, reference[name].state_dict()[key], atol=1e-6, rtol=1e-5)
    assert (state.updates, state.microbatches, state.samples) == (3, 6, 15)


def test_gan_checkpoint_restores_both_optimizers_rng_and_spectral_buffers(tmp_path):
    torch.manual_seed(130)
    modules, optimizers = build()
    state = RunState()
    update_phases(steps(modules), optimizers, state)
    path = tmp_path / "gan.safetensors"
    save_checkpoint(path, modules, optimizers, CheckpointCursor(**asdict(state)), IDENTITY)
    for _ in range(2):
        update_phases(steps(modules), optimizers, state)
    expected_rng = torch.get_rng_state()
    restored, restored_optimizers = build()
    cursor = restore_checkpoint(path, restored, restored_optimizers, IDENTITY)
    restored_state = RunState(**cursor.model_dump())
    for _ in range(2):
        update_phases(steps(restored), restored_optimizers, restored_state)
    assert state == restored_state
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name in modules:
        for key, tensor in modules[name].state_dict().items():
            torch.testing.assert_close(tensor, restored[name].state_dict()[key], atol=0, rtol=0)
        expected, actual = optimizers[name].state_dict(), restored_optimizers[name].state_dict()
        assert actual["param_groups"] == expected["param_groups"]
        for index, terms in expected["state"].items():
            for key, value in terms.items():
                torch.testing.assert_close(value, actual["state"][index][key], atol=0, rtol=0)
    before = copy.deepcopy(restored["generator"].state_dict())
    with pytest.raises(ValueError, match="type"):
        restore_checkpoint(path, restored, {"wrong": restored_optimizers["generator"]}, IDENTITY)
    for key, value in before.items():
        torch.testing.assert_close(value, restored["generator"].state_dict()[key], atol=0, rtol=0)


def test_invalid_second_phase_cannot_partially_step_first_optimizer():
    left, right = nn.Parameter(torch.tensor(2.0)), nn.Parameter(torch.tensor(3.0))
    optimizers = {"generator": torch.optim.SGD([left], lr=0.1), "discriminator": torch.optim.SGD([right], lr=0.1)}
    state = RunState()
    with pytest.raises(FloatingPointError):
        update_phases(
            [StepResult(left.square(), 1, "generator"), StepResult(right * float("nan"), 1, "discriminator")],
            optimizers,
            state,
        )
    assert (left.item(), right.item(), state.updates) == (2.0, 3.0, 0)
    assert left.grad is None and right.grad is None
