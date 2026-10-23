"""Select terminal or notebook tqdm progress while preserving JSON command output."""

import os
import sys
from collections.abc import Iterable, Iterator, Sized
from pathlib import Path
from typing import Literal

from tqdm.auto import tqdm

from unitunner.logging import clear_context, log_status, set_context


def _live_display(notebook: bool) -> bool:
    """Enable widgets or text bars, including shell commands launched by Jupyter."""
    mode = os.environ.get("UNITUNNER_PROGRESS", "auto").lower()
    if mode not in ("auto", "always", "never"):
        raise ValueError("UNITUNNER_PROGRESS must be auto, always or never")
    if mode != "auto":
        return mode == "always"
    if notebook or sys.stderr.isatty() or os.environ.get("JPY_PARENT_PID"):
        return True
    # A kernel without ipywidgets still supports a plain text progress bar.
    ipython = sys.modules.get("IPython")
    get_shell = getattr(ipython, "get_ipython", None)
    return callable(get_shell) and getattr(get_shell(), "kernel", None) is not None


def status(message: str, *, level: Literal["INFO", "WARN", "ERROR"] = "INFO") -> None:
    """Write stage transitions without corrupting active progress bars."""
    log_status(message, level=level)


class Progress:
    """Own a tqdm bar with explicit completed counts and deterministic cleanup."""

    def __init__(
        self,
        label: str,
        total: int | None = None,
        *,
        initial: int = 0,
        unit: str = "item",
        position: int = 0,
        leave: bool = True,
    ) -> None:
        self.total = total
        # Notebook display replaces the literal {bar} with its widget.
        # Limit only terminal bars so wide screens don't stretch accumulation.
        notebook = any(base.__module__ == "tqdm.notebook" for base in tqdm.__mro__)
        bar = "{bar}" if notebook else "{bar:16}"
        self.bar = tqdm(
            total=total,
            initial=initial,
            desc=label,
            unit=unit,
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.2,
            position=position,
            leave=leave,
            disable=not _live_display(notebook),
            colour=("cyan" if position == 0 else "green") if sys.stderr.isatty() else None,
            bar_format="{desc}: {percentage:3.0f}% |"
            + bar
            + "| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            if total is not None
            else None,
        )

    def report(self, completed: int, *, detail: str = "", force: bool = False) -> None:
        """Advance completed work and show metrics without refreshing every item."""
        if detail:
            self.bar.set_postfix_str(detail, refresh=False)
        self.bar.update(completed - self.bar.n)
        if force:
            self.bar.refresh()

    def close(self) -> None:
        """Finish rendering and release the terminal row, including after errors."""
        self.bar.close()


def progress[T](items: Iterable[T], label: str) -> Iterator[T]:
    """Count completed iterations; interrupted encodes never claim completion."""
    owner = Progress(label, len(items) if isinstance(items, Sized) else None)
    completed = 0
    try:
        for item in items:
            source = item if isinstance(item, (str, Path)) else type(item).__name__
            for field in ("path", "high", "source"):
                value = getattr(item, field, None)
                if isinstance(value, (str, Path)):
                    source = value
                    break
            # A consumer's exception is not thrown into this iterator. Retain
            # the active source on early close so the CLI can report it.
            set_context(item=source)
            yield item
            completed += 1
            owner.report(completed, detail=str(getattr(item, "name", "")))
        clear_context("item")
    finally:
        owner.close()
