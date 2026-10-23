"""Compare configured AdamW updates and restored momentum with native PyTorch."""

import copy

import pytest
import torch

from unitunner.config.optimizer import AdamWConfig
from unitunner.config.train import TrainConfig
from unitunner.training.optimizers.adamw import create_adamw
from unitunner.training.optimizers.schedules import apply_learning_rate, learning_rate
from unitunner.training.parameters import ParameterGroup


@pytest.mark.parametrize("backend", ["auto", "single", "foreach"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_configured_adamw_matches_native_and_resumes(backend, dtype):
    actual = torch.nn.Parameter(torch.linspace(-1, 1, 32, dtype=dtype))
    expected = torch.nn.Parameter(actual.detach().clone())
    settings = AdamWConfig(betas=(0.8, 0.95), eps=1e-5, backend=backend)
    optimizer = create_adamw((ParameterGroup("model", (actual,)),), lr=0.002, weight_decay=0.01, config=settings)
    reference = torch.optim.AdamW(
        [expected],
        lr=0.002,
        weight_decay=0.01,
        betas=(0.8, 0.95),
        eps=1e-5,
        fused=False,
        foreach=None if backend == "auto" else backend == "foreach",
    )
    for step in range(4):
        if step == 2:
            saved = copy.deepcopy(optimizer.state_dict())
            optimizer = create_adamw((ParameterGroup("model", (actual,)),), lr=0.0, config=settings)
            optimizer.load_state_dict(saved)
        actual.grad = expected.grad = torch.sin(expected.detach() + step)
        optimizer.step()
        reference.step()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for name in ("exp_avg", "exp_avg_sq"):
        assert optimizer.state[actual][name].dtype == dtype
        torch.testing.assert_close(optimizer.state[actual][name], reference.state[expected][name], atol=0, rtol=0)


def test_explicit_fused_backend_rejects_unsupported_parameters():
    with pytest.raises(ValueError, match="requires CUDA"):
        create_adamw(
            (ParameterGroup("model", (torch.nn.Parameter(torch.ones(2)),)),),
            lr=1e-4,
            config=AdamWConfig(backend="fused"),
        )


def test_learning_rate_policy_applies_to_every_optimizer_phase_and_group():
    parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(3)]
    generator = create_adamw(
        (ParameterGroup("model", tuple(parameters[:1])), ParameterGroup("encoder", tuple(parameters[1:2]), 0.25)),
        lr=0.1,
    )
    discriminator = create_adamw((ParameterGroup("heads", tuple(parameters[2:]), 2.0),), lr=0.3)
    config = TrainConfig(learning_rate=0.01, max_updates=5, warmup_updates=1, lr_schedule="cosine")
    for update in (0, 1, 3, 4):
        rate = apply_learning_rate(config, update, (generator, discriminator))
        assert rate == learning_rate(config, update)
        assert all(
            group["lr"] == rate * group["learning_rate_scale"]
            for optimizer in (generator, discriminator)
            for group in optimizer.param_groups
        )


@pytest.mark.parametrize("invalid", ["empty", "overlap", "duplicate_name", "frozen", "meta"])
def test_group_factory_rejects_invalid_ownership(invalid):
    parameter = torch.nn.Parameter(torch.ones(1))
    groups = (ParameterGroup("model", (parameter,)),)
    if invalid == "empty":
        groups = ()
    elif invalid == "overlap":
        groups += (ParameterGroup("encoder", (parameter,)),)
    elif invalid == "duplicate_name":
        groups += (ParameterGroup("model", (torch.nn.Parameter(torch.ones(1)),)),)
    elif invalid == "frozen":
        parameter.requires_grad_(False)
    else:
        groups = (ParameterGroup("model", (torch.nn.Parameter(torch.ones(1, device="meta")),)),)
    with pytest.raises(ValueError):
        create_adamw(groups, lr=0.01)


def test_scaled_groups_match_native_updates_across_schedule_and_resume():
    parameters = [torch.nn.Parameter(torch.tensor([0.7])), torch.nn.Parameter(torch.tensor([-0.3]))]
    reference_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    config = TrainConfig(learning_rate=0.01, max_updates=6, warmup_updates=2, lr_schedule="linear")
    scales = (0.25, 2.0)

    def build():
        return create_adamw(
            tuple(
                ParameterGroup(str(i), (p,), scale) for i, (p, scale) in enumerate(zip(parameters, scales, strict=True))
            ),
            lr=0.01,
        )

    optimizer = build()
    reference = torch.optim.AdamW(
        [{"params": [p], "lr": 0.01 * scale} for p, scale in zip(reference_parameters, scales, strict=True)],
        weight_decay=1e-4,
    )
    assert [group["lr"] for group in optimizer.param_groups] == [0.0025, 0.02]
    for update in range(6):
        if update == 3:
            state = copy.deepcopy(optimizer.state_dict())
            optimizer = build()
            optimizer.load_state_dict(state)
        apply_learning_rate(config, update, (optimizer,))
        for group, scale in zip(reference.param_groups, scales, strict=True):
            group["lr"] = learning_rate(config, update) * scale
        for actual, expected in zip(parameters, reference_parameters, strict=True):
            actual.grad = expected.grad = expected.detach().cos()
        optimizer.step()
        reference.step()
    for actual, expected in zip(parameters, reference_parameters, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
