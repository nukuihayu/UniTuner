"""Check native Wan solver state ownership and discrete-time expert selection."""

import pytest
import torch

from unituner.models.wan.inference.schedules import build_schedule
from unituner.models.wan.inference.solvers import WanSolver


def test_native_time_truncation_controls_expert_boundary_without_training_offset():
    schedule = build_schedule(3, solver="dpm++", shift=5.0)
    assert schedule.timesteps.tolist() == [1000, 909, 714]
    assert schedule.sigmas[-2] > 0.7141
    assert schedule.high_noise_steps(boundary=0.7141) == (True, True, False)
    assert schedule.high_noise_steps(boundary=0.714) == (True, True, True)
    unipc = build_schedule(3, solver="unipc", shift=1.0)
    assert unipc.timesteps[0] == 999
    assert unipc.sigmas[0] < schedule.sigmas[0]


@pytest.mark.parametrize("solver", ["euler", "dpm++", "unipc"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_solver_owns_one_trajectory_and_rejects_invalid_steps_without_advancing(solver, dtype):
    torch.manual_seed(541)
    state = torch.get_rng_state().clone()
    session = WanSolver(3, solver=solver, shift=5.0)
    assert torch.equal(torch.get_rng_state(), state)
    sample, velocity = torch.randn(1, 16, 2, 4, 4).to(dtype), torch.randn(1, 16, 2, 4, 4).to(dtype)
    original_sample, original_velocity = sample.clone(), velocity.clone()
    with pytest.raises(ValueError, match="matching floating"):
        session.step(velocity.double(), sample)
    assert session.step_index == 0
    result = session.step(velocity, sample)
    assert result.data_ptr() != sample.data_ptr()
    assert torch.equal(sample, original_sample) and torch.equal(velocity, original_velocity)
    if solver != "euler":
        with pytest.raises(ValueError, match="history"):
            session.step(velocity[:, :, :1], sample[:, :, :1])
        assert session.step_index == 1
    result = session.step(velocity, result)
    result = session.step(velocity, result)
    assert torch.isfinite(result).all()
    assert result.dtype == dtype
    assert session.step_index == 3

    with pytest.raises(ValueError, match="exhausted"):
        session.step(velocity, result)
    assert session.step_index == 3


def test_unipc_retains_independent_history_when_caller_reuses_input_storage():
    sample = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 4, 4)
    velocity = torch.full_like(sample, 0.25)
    baseline, reused = WanSolver(3, solver="unipc", shift=5.0), WanSolver(3, solver="unipc", shift=5.0)
    expected = baseline.step(velocity, sample.clone())
    actual = reused.step(velocity, sample)
    sample.add_(100)
    for _ in range(2):
        expected = baseline.step(velocity, expected)
        actual = reused.step(velocity, actual)
    assert torch.equal(actual, expected)
