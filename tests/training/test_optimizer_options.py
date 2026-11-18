"""Compare configured AdamW updates and restored momentum with native PyTorch."""

import copy

import pytest
import torch

from unituner.config.optimizer import AdamWConfig
from unituner.config.train import TrainConfig
from unituner.training.optimizers.adamw import create_adamw
from unituner.training.optimizers.schedules import apply_learning_rate, learning_rate
from unituner.training.parameters import ParameterGroup


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


@pytest.mark.parametrize("precision", ["fp32", "bf16_sr", "int8_sr"])
@pytest.mark.parametrize("factored_2nd", [False, True])
def test_advanced_adamw_state_precision_and_resume(precision, factored_2nd):
    pytest.importorskip("adv_optm")
    settings = AdamWConfig(implementation="adv_optm", state_precision=precision, factored_2nd=factored_2nd)
    parameter = torch.nn.Parameter(torch.linspace(-1, 1, 64).reshape(8, 8))
    reference = torch.nn.Parameter(parameter.detach().clone())

    def build(value):
        return create_adamw((ParameterGroup("model", (value,), 0.5),), lr=0.002, config=settings)

    optimizer = build(parameter)
    uninterrupted = build(reference)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    for step in range(4):
        if step == 2:
            checkpoint = copy.deepcopy(optimizer.state_dict())
            optimizer = build(parameter)
            optimizer.load_state_dict(checkpoint)
        gradient = torch.sin(reference.detach() + step)
        parameter.grad = gradient.clone()
        reference.grad = gradient.clone()
        rng = torch.get_rng_state()
        optimizer.step()
        next_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        uninterrupted.step()
        torch.set_rng_state(next_rng)
        assert torch.isfinite(parameter).all()
        torch.testing.assert_close(parameter, reference, atol=0, rtol=0)

    state = optimizer.state[parameter]
    if factored_2nd:
        assert "exp_avg_sq" not in state
        assert state["mu_v_nmf"].numel() + state["mv_v_nmf"].numel() < parameter.numel()
    else:
        assert "exp_avg_sq" in state
    assert set(optimizer.state_dict()["unituner_rng"]) == {"cpu"}
    if precision == "fp32":
        assert state["exp_avg"].dtype == torch.float32
    elif precision == "bf16_sr":
        assert state["exp_avg"].dtype == torch.bfloat16
    else:
        assert state["exp_avg"].dtype == torch.int8


def test_advanced_adamw_rejects_silent_backend_and_state_fallbacks():
    with pytest.raises(ValueError, match="Compressed optimizer states"):
        AdamWConfig(state_precision="int8_sr")
    with pytest.raises(ValueError, match="Compressed optimizer states"):
        AdamWConfig(factored_2nd=True)
    with pytest.raises(ValueError, match="backend='auto'"):
        AdamWConfig(implementation="adv_optm", backend="fused")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("precision", ["bf16_sr", "int8_sr"])
def test_advanced_adamw_cuda_state_survives_restore(precision):
    pytest.importorskip("adv_optm")
    settings = AdamWConfig(implementation="adv_optm", state_precision=precision, factored_2nd=True)
    original = torch.nn.Parameter(torch.linspace(-1, 1, 64, device="cuda").reshape(8, 8))
    optimizer = create_adamw((ParameterGroup("model", (original,)),), lr=0.001, config=settings)
    original.grad = original.detach().sin()
    optimizer.step()
    restored = torch.nn.Parameter(original.detach().clone())
    resumed = create_adamw((ParameterGroup("model", (restored,)),), lr=0.001, config=settings)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert resumed.state[restored]["exp_avg"].dtype == (torch.bfloat16 if precision == "bf16_sr" else torch.int8)
    original.grad = original.detach().cos()
    restored.grad = original.grad.clone()
    optimizer.step()
    resumed.step()
    torch.testing.assert_close(original, restored, atol=0, rtol=0)


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
        weight_decay=0.01,
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
