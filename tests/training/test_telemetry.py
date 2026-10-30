"""Verify real TensorBoard events and sample-correct accumulated phase metrics."""

from datetime import datetime
from types import SimpleNamespace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn

from unitunner.config.training import OutputConfig
from unitunner.training.contracts import RunState, StepResult
from unitunner.training.engine import update_phases
from unitunner.training.session import TrainingSession
from unitunner.training.telemetry import TrainingTelemetry


def test_phase_metrics_events_and_exception_flush(tmp_path):
    generator = nn.Parameter(torch.tensor(2.0))
    discriminator = nn.Parameter(torch.tensor(3.0))
    optimizers = {
        "generator": torch.optim.SGD([generator], lr=0.1),
        "discriminator": torch.optim.SGD([discriminator], lr=0.2),
    }
    state = RunState(updates=7)
    steps = [
        StepResult(generator.square(), 1, "generator", {"l1": torch.tensor(2.0)}),
        StepResult(discriminator.square(), 1, "discriminator"),
        StepResult(generator.square(), 3, "generator", {"l1": torch.tensor(6.0)}),
        StepResult(discriminator.square(), 3, "discriminator"),
    ]
    results = update_phases(steps, optimizers, state, max_grad_norm=1.0)
    assert results["generator"].metrics["l1"].item() == 5.0
    assert not results["generator"].metrics["l1"].requires_grad
    assert results["generator"].grad_norm.item() == 4.0
    with pytest.raises(RuntimeError, match="training interrupted"), TrainingSession() as session:
        telemetry = session.own(TrainingTelemetry(str(tmp_path), state, name="oil-skin-lora"))
        scalars = telemetry.record(results, optimizers, state, 2.0, torch.device("cpu"))
        raise RuntimeError("training interrupted")
    assert telemetry.directory is not None
    assert telemetry.directory.parent == tmp_path
    assert telemetry.directory.name.startswith("oil-skin-lora_")
    events = EventAccumulator(str(telemetry.directory)).Reload()
    expected = {
        "loss/generator": 4.0,
        "loss/discriminator": 9.0,
        "loss/generator/l1": 5.0,
        "train/generator/grad_norm": 4.0,
        "train/discriminator/lr/0": 0.2,
        "performance/samples_per_second": 2.0,
        "train/samples": 4.0,
    }
    for tag, value in expected.items():
        (event,) = events.Scalars(tag)
        assert event.step == 8
        assert event.value == pytest.approx(value)
        assert event.value == pytest.approx(scalars[tag])


def test_disabled_telemetry_does_not_create_writer():
    telemetry = TrainingTelemetry(None, RunState(), name="oil-skin-lora")
    assert telemetry.writer is None
    telemetry.close()


def test_repeated_runs_have_named_separate_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "unitunner.training.output.datetime", SimpleNamespace(now=lambda: datetime(2026, 10, 21, 15, 30))
    )
    first = TrainingTelemetry(str(tmp_path), RunState(), name="../oil/skin:lora")
    second = TrainingTelemetry(str(tmp_path), RunState(updates=10), name="../oil/skin:lora")
    try:
        assert first.directory is not None and second.directory is not None
        assert first.directory.parent == second.directory.parent == tmp_path
        assert first.directory.name.startswith("oil_skin_lora_")
        assert first.directory != second.directory
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("counts", [(1, 1, 1, 1), (2, 1), (3,)])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_accumulation_uses_actual_samples_and_counts_gan_once(tmp_path, counts, enabled, precision):
    generator = nn.Parameter(torch.tensor(2.0))
    discriminator = nn.Parameter(torch.tensor(3.0))
    optimizers = {
        "generator": torch.optim.SGD([generator], lr=0.1),
        "discriminator": torch.optim.SGD([discriminator], lr=0.2),
    }
    state = RunState(updates=9, samples=20, microbatches=12)

    def steps():
        for index, count in enumerate(counts):
            value = 2.0 + 4.0 * index
            yield StepResult(generator * value, count, "generator", {"l1": torch.tensor(value)})
            yield StepResult(discriminator * (value + 1), count, "discriminator")

    results = update_phases(steps(), optimizers, state, max_grad_norm=1.0, precision=precision)
    expected = sum(count * (2 + 4 * index) for index, count in enumerate(counts)) / sum(counts)
    assert results["generator"].mean_loss.item() == pytest.approx(2 * expected)
    assert results["generator"].grad_norm.item() == pytest.approx(expected)
    assert state.samples == 20 + sum(counts)
    assert state.microbatches == 12 + len(counts)
    telemetry = TrainingTelemetry(str(tmp_path) if enabled else None, state, name="accumulation")
    try:
        scalars = telemetry.record(results, optimizers, state, 2.0, torch.device("cpu"))
    finally:
        telemetry.close()
    assert scalars["loss/generator/l1"] == pytest.approx(expected)
    assert scalars["loss/discriminator"] == pytest.approx(3 * (expected + 1))
    assert scalars["train/update_samples"] == sum(counts)
    assert scalars["train/update_microbatches"] == len(counts)
    assert scalars["performance/samples_per_second"] == sum(counts) / 2
    if enabled:
        events = EventAccumulator(str(telemetry.directory)).Reload()
        for tag, value in scalars.items():
            (event,) = events.Scalars(tag)
            assert event.value == pytest.approx(value)
            assert event.step == 10


@pytest.mark.parametrize("value", ["", " ", "\n"])
def test_blank_log_directory_rejected(value):
    with pytest.raises(ValueError):
        OutputConfig(tensorboard=value)
