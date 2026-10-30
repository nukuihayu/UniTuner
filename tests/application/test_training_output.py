"""Verify execution directory selection, shared names and fixed-run replacement."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from unitunner.cli.main import main
from unitunner.config.loading import load_training_config
from unitunner.config.training import OutputConfig
from unitunner.training.output import (
    prepare_training_output,
    prune_checkpoints,
    publish_latest_weights,
    resume_path,
    weights_path,
)

EXAMPLE = Path(__file__).resolve().parents[2] / "docs/examples/flow.toml"


def test_omitted_directory_shares_execution_name_with_logs_and_tensorboard(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "unitunner.training.output.datetime", SimpleNamespace(now=lambda: datetime(2026, 10, 22, 10, 0))
    )
    config = load_training_config(EXAMPLE).model_copy(
        update={"output": OutputConfig(name="oil-skin-lora", tensorboard=str(tmp_path / "events"))}
    )
    monkeypatch.setattr("unitunner.cli.main.load_training_config", lambda *args, **kwargs: config)
    executions = []

    def run(config, **kwargs):
        from unitunner.training.contracts import RunState
        from unitunner.training.telemetry import TrainingTelemetry

        output = Path(config.output.directory)
        telemetry = TrainingTelemetry(
            config.output.tensorboard, RunState(), name=config.output.name, run_name=config.output._run_name
        )
        try:
            assert telemetry.directory.name == output.name
            assert (output / "logs" / f"{output.name}.jsonl").is_file()
            assert prepare_training_output(config) is config
            executions.append(output)
        finally:
            telemetry.close()
        return 0

    monkeypatch.setattr("unitunner.cli.main.import_module", lambda name: SimpleNamespace(run=run))
    assert main(["train", str(EXAMPLE)]) == 0
    assert main(["train", str(EXAMPLE)]) == 0
    assert executions[0] != executions[1]
    assert all(path.parent == Path("runs") and path.name.startswith("oil-skin-lora_") for path in executions)


def test_fixed_directory_replaces_owned_files_but_resume_preserves_them(tmp_path):
    config = load_training_config(EXAMPLE).model_copy(
        update={"output": OutputConfig(directory=str(tmp_path / "run"), name="model", tensorboard=str(tmp_path / "tb"))}
    )
    output = Path(config.output.directory)
    output.mkdir()
    previews = output / "previews"
    previews.mkdir()
    events = tmp_path / "tb" / "model"
    events.mkdir(parents=True)
    owned = [
        output / "weights/model.safetensors",
        output / "weights/model.encoder.safetensors",
        output / "resume/model.state-00000100.safetensors",
        output / "weights/model-00000100.safetensors",
        output / "weights/model-00000100.encoder.safetensors",
        previews / "step-00000100-00.png",
        events / "events.out.tfevents.old",
    ]
    other = [
        output / "notes.txt",
        output / "other.safetensors",
        previews / "reference.png",
        events / "notes.txt",
        output / "weights/other.safetensors",
        output / "resume/notes.txt",
    ]
    for path in owned + other:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old", encoding="utf-8")
    resumed = prepare_training_output(config, resume=owned[2])
    assert resumed.output.directory == str(output)
    assert all(path.exists() for path in owned + other)
    fresh = prepare_training_output(config)
    assert fresh.output._run_name == "model"
    assert fresh.output.directory == str(output)
    assert not any(path.exists() for path in owned)
    assert all(path.exists() for path in other)


@pytest.mark.parametrize("copy_only", [False, True])
def test_latest_weights_update_with_saves_and_preserve_previous_snapshots(tmp_path, monkeypatch, copy_only):
    config = load_training_config(EXAMPLE).model_copy(
        update={"output": OutputConfig(directory=str(tmp_path), name="model")}
    )
    if copy_only:

        def no_links(*args):
            raise OSError("Hard links unavailable")

        monkeypatch.setattr("unitunner.training.output.os.link", no_links)
    for update in (10, 20):
        for encoder in (False, True):
            path = weights_path(config, update=update, encoder=encoder)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"weights-{update}-{encoder}".encode())
        publish_latest_weights(config, update)
        for encoder in (False, True):
            assert weights_path(config, encoder=encoder).read_bytes() == f"weights-{update}-{encoder}".encode()
    for encoder in (False, True):
        assert weights_path(config, update=10, encoder=encoder).read_bytes() == f"weights-10-{encoder}".encode()


@pytest.mark.parametrize("resume_limit,weight_limit", [(1, 0), (2, 2), (0, 1), (0, 0)])
def test_retention_counts_weight_sets_and_preserves_other_files(tmp_path, resume_limit, weight_limit):
    config = load_training_config(EXAMPLE).model_copy(
        update={
            "output": OutputConfig(
                directory=str(tmp_path),
                name="model[1]",
                max_resume_checkpoints=resume_limit,
                max_weight_checkpoints=weight_limit,
            )
        }
    )
    updates = (1, 2, 100000000)
    for update in updates:
        for path in (
            resume_path(config, update),
            weights_path(config, update=update),
            weights_path(config, update=update, encoder=True),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(update))
    unrelated = [
        tmp_path / "weights/other-00000001.safetensors",
        tmp_path / "resume/other.state-00000001.safetensors",
        tmp_path / "weights/model[1]-00000001.backup.safetensors",
    ]
    for path in unrelated:
        path.write_text("keep")
    publish_latest_weights(config, updates[-1])
    prune_checkpoints(config)
    for update in updates:
        assert resume_path(config, update).exists() == (not resume_limit or update in updates[-resume_limit:])
        for encoder in (False, True):
            assert weights_path(config, update=update, encoder=encoder).exists() == (
                not weight_limit or update in updates[-weight_limit:]
            )
    assert all(path.read_text() == "keep" for path in unrelated)
    for encoder in (False, True):
        assert weights_path(config, encoder=encoder).read_text() == str(updates[-1])


def test_retention_defaults_and_validation():
    from pydantic import ValidationError

    assert OutputConfig().max_resume_checkpoints == 1
    assert OutputConfig().max_weight_checkpoints == 0
    for key in ("max_resume_checkpoints", "max_weight_checkpoints"):
        with pytest.raises(ValidationError):
            OutputConfig.model_validate({key: -1})
