"""Verify readable terminal output without losing machine-readable diagnostics."""

import io
import json
import sys
from os import terminal_size

import pytest

from unituner.logging import configure_logging, get_logger, log_table, logging_session, publish_result
from unituner.progress import Progress, status
from unituner.terminal import cell_width, plain, table_lines


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_terminal_colors_stages_values_and_success_without_redundant_info(tmp_path, monkeypatch):
    stream = Terminal()
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    path = tmp_path / "events.jsonl"
    with logging_session():
        configure_logging(log_file=path)
        status("[train] Loading model, rank=32 lr=1e-4")
        status("[checkpoint] Saved weights in 2.5s", complete=True)
        get_logger().warning("[cache] 2 missing records")
    rendered = stream.getvalue()
    assert "\x1b[94m[train]" in rendered
    assert "\x1b[96m32" in rendered and "\x1b[96m1e-4" in rendered
    assert "\x1b[92m[checkpoint]" in rendered
    assert "[WARNING] [cache]" in plain(rendered)
    assert " INFO " not in rendered
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["stage"] == "train"
    assert events[1]["event"] == "complete"
    assert all("\x1b" not in json.dumps(event) for event in events)


@pytest.mark.parametrize("width", [40, 80, 160])
def test_tables_fit_terminal_including_wide_names_and_preserve_raw_file_values(width, tmp_path, monkeypatch):
    stream = Terminal()
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.setattr("unituner.logging.get_terminal_size", lambda **kwargs: terminal_size((width, 24)))
    path = tmp_path / "events.jsonl"
    name = "模型.transformer_blocks.123.attn.to_out.0" * 3
    with logging_session():
        configure_logging(log_file=path)
        log_table(
            "Adapter spectrum", ("LAYER", "RANK", "R_STABLE", "STATUS"), [(name, 32, 3.25, "EXACT")], right=(1, 2)
        )
        log_table("Empty results", ("LAYER", "STATUS"), [])
    assert all(cell_width(line) <= width for line in stream.getvalue().splitlines())
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["table"]["rows"][0] == [name, 32, 3.25, "EXACT"]
    assert events[1]["table"]["rows"] == []


@pytest.mark.parametrize("setting", [("NO_COLOR", ""), ("TERM", "dumb")])
def test_color_opt_out_applies_to_records_tables_and_live_progress(monkeypatch, setting):
    stream = Terminal()
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv(*setting)
    monkeypatch.setenv("UNITUNER_PROGRESS", "always")
    with logging_session():
        status("[train] Loading 32 layers")
        log_table("Results", ("LAYER", "STATUS"), [("block.0", "EXACT")])
        owner = Progress("[train] updates", 1)
        owner.report(1, detail="loss=0.25", force=True)
        owner.close()
    assert "\x1b[9" not in stream.getvalue()
    assert "\x1b[0m" not in stream.getvalue()
    assert "100%" in stream.getvalue()


def test_repeated_info_is_coalesced_but_all_warnings_and_file_events_survive(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    with logging_session():
        configure_logging(log_file=path)
        for _ in range(5):
            status("[cache] Waiting for input")
        for _ in range(2):
            status("[cache] Missing source", level="WARN")
        for _ in range(3):
            status("[cache] Finishing")
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err.count("Waiting for input") == 1
    assert captured.err.count("Missing source") == 2
    assert "repeated 4 more" in captured.err and "repeated 2 more" in captured.err
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 10


@pytest.mark.parametrize("interactive", [False, True])
def test_result_stdout_is_json_for_pipes_and_readable_for_terminals(interactive, monkeypatch):
    out = Terminal() if interactive else io.StringIO()
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    with logging_session():
        publish_result({"valid": True, "family": "qwen-image"}, stage="validate", message="Configuration valid")
    assert "[validate] Configuration valid" in err.getvalue()
    if interactive:
        assert not out.getvalue()
    else:
        assert json.loads(out.getvalue()) == {"valid": True, "family": "qwen-image"}
    assert "\x1b" not in err.getvalue()


def test_narrow_spectral_table_uses_labeled_rows():
    lines = table_lines(
        ("LAYER", "RANK", "R_EFF", "R_STABLE", "K_99%", "NORM_F", "STATUS"),
        [("block.0.attn", 32, 3.5, 2.5, 4, 1.5, "EXACT")],
        width=40,
        color=False,
    )
    assert "LAYER: block.0.attn" in lines and "STATUS: EXACT" in lines
    assert all(cell_width(line) <= 40 for line in lines)


@pytest.mark.parametrize("state,code", [("EXACT", 92), ("ZERO", 93), ("ERROR", 91), ("SKIPPED_BUDGET", 90)])
def test_status_colors_survive_narrow_layouts_and_clipping(state, code):
    narrow = table_lines(
        ("LAYER", "RANK", "R_EFF", "R_STABLE", "K_99%", "NORM_F", "STATUS"),
        [("block.0.attn", 32, 3.5, 2.5, 4, 1.5, state)],
        width=40,
        color=True,
    )
    assert f"\x1b[{code}mSTATUS: {state}\x1b[0m" in narrow
    clipped = table_lines(("LAYER", "STATUS"), [("a", state)], width=14, color=True)
    assert any(f"\x1b[{code}m" in line for line in clipped[2:-1])
    assert all(cell_width(line) <= 14 for line in clipped)
