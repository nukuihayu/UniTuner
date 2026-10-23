"""Keep terminal summaries bounded without dropping persisted update metrics."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from unitunner.logging import logging_session, start_training_log
from unitunner.progress import Progress
from unitunner.training.progress import TrainingProgress


def test_notebook_bash_child_shows_progress_without_json_summaries():
    """Exercise the pipe-backed subprocess used by a notebook bash cell."""
    script = """
from unitunner.training.progress import TrainingProgress
display = TrainingProgress(1)
display.report(1, 1, {
    'loss/main': 0.25,
    'performance/update_seconds': 1.0,
    'train/main/lr/0': 0.00005,
})
display.close()
"""
    result = subprocess.run(
        ["bash", "-c", 'exec "$1" -c "$2"', "notebook-test", sys.executable, script],
        env={**os.environ, "JPY_PARENT_PID": str(os.getpid()), "UNITUNNER_PROGRESS": "auto"},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert not result.stdout
    assert "100%" in result.stderr and "loss=0.25" in result.stderr


@pytest.mark.parametrize(
    ("mode", "parent", "kernel", "enabled"),
    [
        ("auto", "", False, False),
        ("auto", "123", False, True),
        ("auto", "", True, True),
        ("always", "", False, True),
        ("never", "123", True, False),
    ],
)
def test_non_tty_progress_in_notebook_and_redirected_commands(monkeypatch, capsys, mode, parent, kernel, enabled):
    """Render a text fallback for notebook shells without breaking redirected JSON."""
    from tqdm.std import tqdm

    monkeypatch.setenv("UNITUNNER_PROGRESS", mode)
    monkeypatch.setenv("JPY_PARENT_PID", parent)
    monkeypatch.setattr("unitunner.progress.tqdm", tqdm)
    shell = SimpleNamespace(kernel=object()) if kernel else None
    monkeypatch.setitem(sys.modules, "IPython", SimpleNamespace(get_ipython=lambda: shell))
    display = Progress("updates", 2)
    try:
        assert display.bar.disable is not enabled
        display.report(2, detail="loss=0.25", force=True)
    finally:
        display.close()
    captured = capsys.readouterr()
    assert not captured.out
    assert ("100%" in captured.err and "loss=0.25" in captured.err) is enabled


@pytest.mark.parametrize("live", [False, True])
def test_training_display_throttles_console_but_keeps_all_file_metrics(tmp_path, monkeypatch, capsys, live):
    reports = []

    class Display:
        def __init__(self, *args, **kwargs):
            self.bar = SimpleNamespace(disable=not live)

        def report(self, update, **kwargs):
            reports.append((update, kwargs["detail"]))

        def close(self):
            pass

    monkeypatch.setattr("unitunner.training.progress.Progress", Display)
    times = iter((0.0, 10.0, 59.0, 60.0, 61.0))
    monkeypatch.setattr("unitunner.training.progress.monotonic", lambda: next(times))
    scalars = {
        "loss/generator": 4.5,
        "loss/discriminator": 2.0,
        "performance/update_seconds": 10.0,
        "train/generator/lr/0": 0.00005,
        "memory/allocated_gib": 14.0,
        "memory/peak_allocated_gib": 51.0,
    }
    with logging_session():
        path = start_training_log(str(tmp_path), "model", run_name="model")
        display = TrainingProgress(15, initial=10)
        for update in range(11, 16):
            display.report(update, update * 4, scalars)
        display.close()
    captured = capsys.readouterr()
    summaries = [json.loads(line)["update"] for line in captured.out.splitlines()]
    assert summaries == ([] if live else [11, 14, 15])
    assert captured.err.count("[train] update=") == (0 if live else 3)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    updates = [event for event in events if "metrics" in event]
    assert len(updates) == 5
    assert all(event["metrics"] == scalars for event in updates)
    assert [update for update, _ in reports] == list(range(11, 16))
    assert all("D=2" in detail and "peak=51.0GiB" in detail for _, detail in reports)
