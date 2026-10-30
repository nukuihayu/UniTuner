"""Exercise unified CLI failures, structured logs and logging isolation."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from unituner.cli.main import main
from unituner.config.loading import load_training_config
from unituner.logging import configure_logging, get_logger, logging_session, operation_context, report_error
from unituner.progress import progress, status

EXAMPLE = Path(__file__).resolve().parents[2] / "docs/examples/flow.toml"


@pytest.mark.parametrize(
    "error,code,hint",
    [
        (RuntimeError("kernel failed"), 1, "RuntimeError"),
        (ValueError("invalid topology"), 2, "ValueError"),
        (FileNotFoundError(2, "missing model", "model.safetensors"), 2, "working directory"),
        (ModuleNotFoundError("missing backend", name="transformers"), 2, "uv sync"),
        (FloatingPointError("nonfinite loss"), 1, "fp16_loss_scale"),
        (RuntimeError("CUDA out of memory"), 1, "Reduce batch size"),
    ],
)
def test_failures_include_phase_context_trace_and_json_file(tmp_path, monkeypatch, capsys, error, code, hint):
    def run(config, **kwargs):
        status("[train] decoding VAE")
        with operation_context(update=17, source="bad-image.png"):
            raise error

    monkeypatch.setattr("unituner.cli.main.import_module", lambda name: SimpleNamespace(run=run))
    run_dir = tmp_path / "run"
    assert main(["train", str(EXAMPLE), "--output.directory", str(run_dir)]) == code
    (path,) = (run_dir / "logs").glob("*.jsonl")
    captured = capsys.readouterr()
    assert not captured.out
    assert "ERROR" in captured.err and hint in captured.err
    assert "decoding VAE" in captured.err and "command=train" in captured.err
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    failure = events[-1]
    assert failure["exception"]["type"] == type(error).__name__
    assert "bad-image.png" in failure["exception"]["traceback"]
    assert "update=17" in failure["exception"]["traceback"]
    assert failure["context"]["config"] == str(EXAMPLE)
    assert failure["context"]["source"] == "bad-image.png"
    assert failure["context"]["update"] == "17"
    assert "\x1b" not in path.read_text(encoding="utf-8")
    # Closed files can be renamed immediately, including on Windows.
    path.rename(tmp_path / "closed.jsonl")


def test_nested_exception_chain_survives_logging(tmp_path, monkeypatch, capsys):
    def run(config, **kwargs):
        try:
            raise OSError("disk read failed")
        except OSError as cause:
            raise RuntimeError("Cannot load weights") from cause

    monkeypatch.setattr("unituner.cli.main.import_module", lambda name: SimpleNamespace(run=run))
    run_dir = tmp_path / "run"
    assert main(["--log-level", "INFO", "train", str(EXAMPLE), "--output.directory", str(run_dir)]) == 1
    (path,) = (run_dir / "logs").glob("*.jsonl")
    error = capsys.readouterr().err
    assert "disk read failed" in error and "Cannot load weights" in error
    event = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert "direct cause" in event["exception"]["traceback"]


def test_failed_cache_consumer_retains_active_filename(tmp_path, monkeypatch, capsys):
    def run(config, **kwargs):
        for _ in progress([Path("broken.png")], "encoding"):
            raise ValueError("cannot decode image")

    monkeypatch.setattr("unituner.cli.main.import_module", lambda name: SimpleNamespace(run=run))
    assert main(["train", str(EXAMPLE), "--output.directory", str(tmp_path / "run")]) == 2
    assert "item=broken.png" in capsys.readouterr().err


def test_configuration_errors_never_resurrect_secret_input(tmp_path, capsys):
    path = tmp_path / "safe.jsonl"
    with logging_session():
        configure_logging(level="DEBUG", log_file=path)
        try:
            load_training_config(EXAMPLE, overrides={"train": {"learning_rate": "secret-value"}})
        except ValueError as error:
            assert report_error(error) == 2
        else:
            pytest.fail("Expected configuration validation to fail")
    captured = capsys.readouterr()
    content = path.read_text(encoding="utf-8")
    assert "secret-value" not in captured.err + content
    assert "train.learning_rate" in captured.err
    assert json.loads(content.splitlines()[-1])["exception"]["traceback"] is None


def test_console_level_and_sessions_do_not_duplicate_or_change_host_logging(tmp_path, monkeypatch, capsys):
    host = logging.getLogger()
    handlers = host.handlers[:]
    app = get_logger()
    application_handlers = app.handlers[:]

    def run(config, **kwargs):
        status("hidden info")
        get_logger("testing").debug("file-only diagnostic")
        get_logger("testing").warning("visible warning")
        return 0

    monkeypatch.setattr("unituner.cli.main.import_module", lambda name: SimpleNamespace(run=run))
    run_dir = tmp_path / "run"
    for _ in range(2):
        assert main(["train", str(EXAMPLE), "--log-level", "WARNING", "--output.directory", str(run_dir)]) == 0
    error = capsys.readouterr().err
    assert "hidden info" not in error and "file-only diagnostic" not in error
    assert error.count("visible warning") == 2
    paths = list((run_dir / "logs").glob("*.jsonl"))
    assert len(paths) == 1
    assert sum(path.read_text(encoding="utf-8").count("file-only diagnostic") for path in paths) == 1
    assert host.handlers == handlers and app.handlers == application_handlers


def test_interrupt_and_argument_errors_have_clean_exit_semantics(tmp_path, monkeypatch, capsys):
    def run(config, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("unituner.cli.main.import_module", lambda name: SimpleNamespace(run=run))
    assert main(["train", str(EXAMPLE), "--output.directory", str(tmp_path / "run")]) == 130
    assert "Traceback" not in capsys.readouterr().err
    with pytest.raises(SystemExit) as result:
        main(["train", "--unknown-option"])
    assert result.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out and "ERROR" in captured.err and "usage:" in captured.err
