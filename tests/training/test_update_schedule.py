"""Pin save and preview events across final updates and interrupted continuation."""

from pathlib import Path

import pytest
import torch

from unitunner.checkpoints.restore import restore_checkpoint
from unitunner.checkpoints.schema import build_resume_identity
from unitunner.config.loading import load_training_config
from unitunner.training.contracts import RunState, StepResult
from unitunner.training.engine import run_updates
from unitunner.training.optimizers.adamw import create_adamw
from unitunner.training.output import resume_path
from unitunner.training.parameters import ParameterGroup
from unitunner.training.update_schedule import UpdateSchedule

EXAMPLE = Path(__file__).parents[2] / "docs/examples/full.toml"


@pytest.mark.parametrize("preview,at_start", [(False, False), (True, False), (True, True)])
def test_cadence_final_update_and_initial_preview(preview, at_start):
    schedule = UpdateSchedule(5, 2, 3 if preview else None, at_start)
    assert [i for i in range(7) if schedule.save_due(i)] == [2, 4, 5]
    assert [i for i in range(7) if schedule.preview_due(i)] == ([3, 5] if preview else [])
    assert schedule.initial_preview_due(0) == at_start
    assert not any(schedule.initial_preview_due(i) for i in (1, 2, 5))


def test_shared_loop_publication_order_and_resume_match_continuous_training(tmp_path):
    config = load_training_config(
        EXAMPLE,
        overrides={
            "train": {
                "parameter_mode": "full",
                "adapter": None,
                "learning_rate": 0.01,
                "max_updates": 5,
                "warmup_updates": 1,
                "lr_schedule": "cosine",
                "optimizer": {"betas": [0.8, 0.95], "eps": 1e-6},
            },
            "execution": {"precision": "fp32"},
            "preview": {"at_start": True, "every_updates": 3, "samples": [{"prompt": ""}]},
            "output": {"directory": str(tmp_path / "continuous"), "save_every_updates": 2, "max_resume_checkpoints": 0},
        },
    )
    identity = build_resume_identity(config, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64)

    def run(current_config, *, resume=None):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.5)
        optimizer = create_adamw(
            (ParameterGroup("model", tuple(model.parameters())),),
            lr=config.train.learning_rate,
            config=config.train.optimizer,
        )
        state = RunState()
        if resume is not None:
            cursor = restore_checkpoint(resume, {"model": model}, optimizer, identity)
            state = RunState(cursor.updates, cursor.microbatches, cursor.samples)
        events = []

        def steps():
            yield StepResult(model(torch.ones(1, 1)).square().mean(), 1)

        def weights(update):
            events.append(("weights", update))

        def preview(update):
            if update in (0, 3):
                assert not resume_path(current_config, update).is_file()
            else:
                assert resume_path(current_config, update).is_file()
            events.append(("preview", update))

        run_updates(
            current_config,
            steps,
            optimizer=optimizer,
            modules={"model": model},
            state=state,
            identity=identity,
            device=torch.device("cpu"),
            weights_writer=weights,
            preview_writer=preview,
        )
        return model, optimizer, state, events

    continuous, optimizer, state, events = run(config)
    assert events == [("preview", 0), ("weights", 2), ("preview", 3), ("weights", 4), ("weights", 5), ("preview", 5)]
    resumed_config = config.model_copy(
        update={"output": config.output.model_copy(update={"directory": str(tmp_path / "resumed")})}
    )
    resumed, resumed_optimizer, resumed_state, resumed_events = run(resumed_config, resume=resume_path(config, 2))
    assert state == resumed_state == RunState(updates=5, microbatches=5, samples=5)
    assert resumed_events == events[2:]
    torch.testing.assert_close(resumed.weight, continuous.weight, atol=0, rtol=0)
    for name in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(
            resumed_optimizer.state[resumed.weight][name], optimizer.state[continuous.weight][name], atol=0, rtol=0
        )


@pytest.mark.parametrize("phases", [("main",), ("generator", "discriminator")])
@pytest.mark.parametrize("fail", [False, True])
def test_shared_loop_closes_steps_and_preserves_optimizer_roles(tmp_path, phases, fail):
    config = load_training_config(
        EXAMPLE,
        overrides={
            "train": {"max_updates": 1},
            "execution": {"precision": "fp32"},
            "preview": {"at_start": False, "every_updates": 1, "samples": [{"prompt": ""}]},
            "output": {"directory": str(tmp_path), "save_every_updates": 1},
        },
    )
    identity = build_resume_identity(config, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64)
    modules = {phase: torch.nn.Linear(1, 1, bias=False) for phase in phases}
    optimizers = {
        phase: torch.optim.SGD(module.parameters(), lr=0.1, momentum=0.9) for phase, module in modules.items()
    }
    optimizer = optimizers["main"] if phases == ("main",) else optimizers
    state = RunState()
    events = []

    def steps():
        try:
            for phase, module in modules.items():
                yield StepResult(module(torch.ones(1, 1)).square().mean(), 1, phase="unknown" if fail else phase)
        finally:
            events.append("closed")

    def preview(update):
        assert update == 1
        assert all(parameter.grad is None for module in modules.values() for parameter in module.parameters())
        assert resume_path(config, update).is_file()
        events.append("preview")

    def run():
        return run_updates(
            config,
            steps,
            optimizer=optimizer,
            modules=modules,
            state=state,
            identity=identity,
            device=torch.device("cpu"),
            weights_writer=lambda update: events.append("weights"),
            preview_writer=preview,
        )

    if fail:
        with pytest.raises(ValueError, match="Unexpected optimizer phase"):
            run()
        assert events == ["closed"]
        assert state == RunState()
        assert not resume_path(config, 1).exists()
        assert all(parameter.grad is None for module in modules.values() for parameter in module.parameters())
    else:
        assert run() == RunState(updates=1, microbatches=1, samples=1)
        assert events == ["closed", "weights", "preview"]
        expected = {phase: module.weight.detach().clone() for phase, module in modules.items()}
        for module in modules.values():
            with torch.no_grad():
                module.weight.zero_()
        cursor = restore_checkpoint(resume_path(config, 1), modules, optimizer, identity)
        assert cursor.updates == 1
        for phase, module in modules.items():
            torch.testing.assert_close(module.weight, expected[phase], atol=0, rtol=0)
            assert "momentum_buffer" in optimizers[phase].state[module.weight]
