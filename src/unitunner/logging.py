"""Own tqdm-compatible application logging and contextual failure reports."""

import json
import logging
import os
import re
import sys
import textwrap
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from shutil import get_terminal_size
from typing import Any, TextIO

from tqdm.auto import tqdm

_context: ContextVar[dict[str, Any] | None] = ContextVar("unitunner_log_context", default=None)
_LOGGER = "unitunner"


def set_context(**fields: Any) -> None:
    """Attach explicit diagnostic fields without retaining tensors or config values."""
    _context.set((_context.get() or {}) | {key: str(value) for key, value in fields.items() if value is not None})


def clear_context(*fields: str) -> None:
    """Remove completed-operation fields without discarding command diagnostics."""
    _context.set({key: value for key, value in (_context.get() or {}).items() if key not in fields})


def _trace(record: logging.LogRecord, *, full: bool) -> str | None:
    if record.exc_info is None or record.exc_info[1] is None:
        return None
    error = record.exc_info[1]
    # Configuration diagnostics already omit raw values; never resurrect the
    # suppressed Pydantic/TOML exception via a debug traceback or file handler.
    if getattr(error, "hide_traceback", False):
        return None
    return "".join(
        traceback.TracebackException.from_exception(error, limit=None if full else -8, capture_locals=False).format()
    ).rstrip()


class ConsoleHandler(logging.Handler):
    """Render wrapped records beside live tqdm bars, using the current stderr."""

    def emit(self, record: logging.LogRecord) -> None:
        if not getattr(record, "console", True):
            return
        timestamp = datetime.fromtimestamp(record.created).astimezone().strftime("%H:%M:%S")
        level = record.levelname
        plain = f"{timestamp} {level:<7} "
        prefix = plain
        if sys.stderr.isatty() and "NO_COLOR" not in os.environ:
            color = {"DEBUG": 90, "INFO": 36, "WARNING": 33, "ERROR": 31, "CRITICAL": 31}.get(level, 0)
            prefix = f"\033[2m{timestamp}\033[0m \033[{color}m{level:<7}\033[0m "
        width = max(40, get_terminal_size(fallback=(120, 24)).columns)
        messages = record.getMessage().splitlines() or [""]
        context = getattr(record, "context", _context.get() or {})
        if record.levelno >= logging.ERROR and context:
            messages += ["Context: " + " | ".join(f"{key}={value}" for key, value in context.items())]
        trace = _trace(record, full=getattr(self, "full_trace", False))
        if trace:
            messages.extend(trace.splitlines())
        first = True
        for message in messages:
            for line in textwrap.wrap(
                message,
                width=width - len(plain),
                replace_whitespace=False,
                drop_whitespace=True,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]:
                tqdm.write((prefix if first else " " * len(plain)) + line, file=sys.stderr)
                first = False
        sys.stderr.flush()


class JsonFileHandler(logging.Handler):
    """Append timestamped structured records and full exception chains as JSONL."""

    def __init__(self, path: Path, *, mode: str = "a") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream: TextIO = path.open(mode, encoding="utf-8")
        super().__init__(logging.DEBUG)

    def emit(self, record: logging.LogRecord) -> None:
        event = {
            "time": datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "context": getattr(record, "context", _context.get() or {}).copy(),
            "source": {"path": record.pathname, "line": record.lineno, "function": record.funcName},
        }
        if hasattr(record, "metrics"):
            event["metrics"] = record.metrics
        if record.exc_info and record.exc_info[1]:
            error = record.exc_info[1]
            event["exception"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": _trace(record, full=True),
            }
        if getattr(self, "failed", False):
            return
        try:
            self.stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.stream.flush()
        except OSError as error:
            self.failed = True
            ConsoleHandler().emit(
                logging.LogRecord(_LOGGER, logging.ERROR, __file__, 0, f"Cannot write log file: {error}", (), None)
            )

    def close(self) -> None:
        try:
            self.stream.close()
        except OSError:
            pass  # A failed log disk must not mask the original application error.
        super().close()


def get_logger(name: str | None = None) -> logging.Logger:
    """Use a package logger without replacing the host notebook/root handlers."""
    parent = logging.getLogger(_LOGGER)
    if not parent.handlers:
        parent.addHandler(ConsoleHandler())
        parent.setLevel(logging.INFO)
        parent.propagate = False
    return parent if name is None else logging.getLogger(f"{_LOGGER}.{name}")


def configure_logging(*, level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure this invocation; file logs retain DEBUG records independently."""
    logger = get_logger()
    if log_file is not None:
        logger.addHandler(JsonFileHandler(log_file))
    logger.setLevel(logging.DEBUG if log_file is not None else level)
    for handler in logger.handlers:
        if isinstance(handler, ConsoleHandler):
            handler.setLevel(level)
            handler.full_trace = level == "DEBUG"


def start_training_log(directory: str, name: str, *, run_name: str | None = None, append: bool = False) -> Path:
    """Create a distinct diagnostic file in this training run's output directory."""
    safe_name = re.sub(r"[^\w.-]+", "_", name).strip("._-") or "train"
    run_name = run_name or f"{safe_name}_{datetime.now():%Y%m%d-%H%M%S-%f}"
    path = Path(directory) / "logs" / f"{run_name}.jsonl"
    logger = get_logger()
    logger.addHandler(JsonFileHandler(path, mode="a" if append else "w"))
    logger.setLevel(logging.DEBUG)
    logger.debug("Runtime Python=%s platform=%s cwd=%s", sys.version.split()[0], sys.platform, Path.cwd())
    return path


@contextmanager
def logging_session() -> Iterator[None]:
    """Restore all package handlers/context after a CLI call, including failures."""
    logger = logging.getLogger(_LOGGER)
    previous = logger.handlers[:], logger.level, logger.propagate
    token = _context.set({})
    logger.handlers = [ConsoleHandler()]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers, level, logger.propagate = previous
        logger.setLevel(level)
        _context.reset(token)


def log_status(message: str, *, level: str = "INFO") -> None:
    """Route existing stage messages through standard logging and retain the phase."""
    if level in ("INFO", "DEBUG") and not message.startswith("[train] update="):
        set_context(stage=message)
    get_logger().log(
        logging.WARNING if level == "WARN" else getattr(logging, level),
        message,
        extra={"context": (_context.get() or {}).copy()},
        stacklevel=3,
    )


@contextmanager
def operation_context(**fields: Any) -> Iterator[None]:
    """Add source/update notes to failures before restoring borrowed context."""
    token = _context.set(
        (_context.get() or {}) | {key: str(value) for key, value in fields.items() if value is not None}
    )
    try:
        yield
    except Exception as error:
        error.__dict__.setdefault("unitunner_context", (_context.get() or {}).copy())
        error.add_note("Operation: " + " | ".join(f"{key}={value}" for key, value in fields.items()))
        raise
    finally:
        _context.reset(token)


def report_error(error: Exception) -> int:
    """Report application failures once, retaining exception chains and actionable hints."""
    hints = []
    if isinstance(error, FileNotFoundError):
        hints.append("Check the configured input path and current working directory.")
    elif isinstance(error, PermissionError):
        hints.append("Check permissions and whether another process has the file open.")
    elif isinstance(error, ImportError):
        hints.append(
            "Install the required extras in the environment running unitunner (uv sync --extra text --extra media)."
        )
    elif isinstance(error, FloatingPointError):
        hints.append("Check nonfinite inputs/losses and learning rate; for FP16 also check execution.fp16_loss_scale.")
    detail = str(error).lower()
    if "cache" in detail and any(word in detail for word in ("digest", "identity", "manifest", "rebuild")):
        hints.append(
            "Rebuild the cache with the matching configuration before training; check the reported source/group."
        )
    if "outofmemory" in type(error).__name__.lower() or "out of memory" in str(error).lower():
        hints.append("Reduce batch size/resolution, enable recomputation, and inspect CUDA memory/other GPU processes.")
        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.cuda.is_initialized():
                device = torch.cuda.current_device()
                set_context(
                    cuda_device=device,
                    cuda_allocated_gib=f"{torch.cuda.memory_allocated(device) / 2**30:.2f}",
                    cuda_reserved_gib=f"{torch.cuda.memory_reserved(device) / 2**30:.2f}",
                    cuda_peak_gib=f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f}",
                )
        except Exception:
            pass  # CUDA may be unusable after the original failure.
    message = f"{type(error).__name__}: {error}"
    for hint in hints:
        message += f"\nHint: {hint}"
    get_logger("cli").error(
        message,
        exc_info=(type(error), error, error.__traceback__),
        extra={"context": (_context.get() or {}) | getattr(error, "unitunner_context", {})},
    )
    return 2 if isinstance(error, (ValueError, OSError, ImportError)) else 1
