"""Check full/adapter updates against concatenated-batch optimizer mathematics."""

import copy

import pytest
import torch
from torch import nn

from unituner.adapters.attachment import AdapterAttachment
from unituner.adapters.lora import LoRA
from unituner.training.contracts import RunState, StepResult
from unituner.training.engine import update_phase
from unituner.training.losses.flow_matching import flow_matching_loss
from unituner.training.session import TrainingSession


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weighted", [False, True])
def test_flow_loss_matches_elementwise_reference_and_input_gradients(dtype, weighted):
    torch.manual_seed(83)
    prediction = torch.randn(3, 2, 5, 7, dtype=dtype, requires_grad=True)
    target = torch.randn_like(prediction).requires_grad_()
    weights = torch.rand(3, requires_grad=True) if weighted else None
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    errors = (prediction.to(compute_dtype) - target.to(compute_dtype)).square().flatten(1).mean(1)
    expected = errors.mean() if weights is None else (errors * weights.to(compute_dtype)).mean()
    actual = flow_matching_loss(prediction, target, weights)
    torch.testing.assert_close(actual, expected)
    inputs = (prediction, target) if weights is None else (prediction, target, weights)
    for a, b in zip(torch.autograd.grad(actual, inputs), torch.autograd.grad(expected, inputs), strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("adapter", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_uneven_accumulation_matches_three_dense_updates(adapter, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(82)
    model = nn.Linear(6, 4, dtype=torch.float64, device=device)
    reference = copy.deepcopy(model)
    with TrainingSession() as session:
        if adapter:
            model.requires_grad_(False)
            reference.requires_grad_(False)
            branch = LoRA(6, 4, rank=2, alpha=2.0, dtype=torch.float64, device=torch.device(device))
            with torch.no_grad():
                branch.up.normal_(0, 0.2)
            ref_branch = copy.deepcopy(branch)
            session.own(AdapterAttachment(model, branch))
            session.own(AdapterAttachment(reference, ref_branch))
            parameters, ref_parameters = list(branch.parameters()), list(ref_branch.parameters())
        else:
            parameters, ref_parameters = list(model.parameters()), list(reference.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=0.02, weight_decay=0.01)
        ref_optimizer = torch.optim.AdamW(ref_parameters, lr=0.02, weight_decay=0.01)
        state = RunState()
        for _ in range(3):
            inputs = torch.randn(5, 6, dtype=torch.float64, device=device)
            target = torch.randn(5, 4, dtype=torch.float64, device=device)

            def steps(inputs=inputs, target=target):
                for start, end in [(0, 2), (2, 5)]:
                    yield StepResult(flow_matching_loss(model(inputs[start:end]), target[start:end]), end - start)

            update_phase(steps(), optimizer, state, max_grad_norm=1.0)
            ref_optimizer.zero_grad(set_to_none=True)
            flow_matching_loss(reference(inputs), target).backward()
            nn.utils.clip_grad_norm_(ref_parameters, 1.0, error_if_nonfinite=True)
            ref_optimizer.step()
            for actual, expected in zip(parameters, ref_parameters, strict=True):
                torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
        assert (state.updates, state.microbatches, state.samples) == (3, 6, 15)


def test_nonfinite_phase_does_not_update_or_advance():
    parameter = nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    state = RunState()
    with pytest.raises(FloatingPointError, match="non-finite"):
        update_phase([StepResult(parameter * float("nan"), 1)], optimizer, state)
    assert parameter.item() == 2.0
    assert state.updates == 0
    assert parameter.grad is None


def test_unrelated_loss_cannot_count_as_optimizer_update():
    parameter = nn.Parameter(torch.tensor(2.0))
    unrelated = nn.Parameter(torch.tensor(3.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    state = RunState()
    with pytest.raises(ValueError, match="no optimizer gradients"):
        update_phase([StepResult(unrelated.square(), 1)], optimizer, state)
    assert state.updates == 0
    assert parameter.item() == 2.0


def test_nonfinite_loss_with_finite_derivative_is_rejected():
    parameter = nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    state = RunState()
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        update_phase([StepResult(parameter + float("nan"), 1)], optimizer, state)
    assert state.updates == 0
    assert parameter.item() == 2.0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_loss_is_accumulated_in_fp32(dtype):
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.0)
    value = 40000.0 if dtype == torch.float16 else 1.0
    loss = (parameter * value).to(dtype)
    result = update_phase(
        [StepResult(loss, 2), StepResult((parameter * value).to(dtype), 3)],
        optimizer,
        RunState(),
        max_grad_norm=1.0,
    )
    assert result.mean_loss.dtype == torch.float32
    assert result.mean_loss.item() == loss.detach().float().item()


def test_missing_component_cannot_silently_dilute_metric():
    parameter = nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    state = RunState()
    steps = [
        StepResult(parameter.square(), 1, metrics={"l1": torch.tensor(4.0)}),
        StepResult(parameter.square(), 3),
    ]
    with pytest.raises(ValueError, match="metric keys"):
        update_phase(steps, optimizer, state)
    assert state.updates == 0
    assert parameter.item() == 2.0
    assert parameter.grad is None
