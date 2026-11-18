"""Render compact terminal records and width-aware tables without extra dependencies."""

import os
import re
import unicodedata
from collections.abc import Sequence
from typing import TextIO

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_NUMBER = re.compile(r"(?<![\w./\\])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:%|GiB|MiB|KiB|GB|MB|ms|s)?(?![\w.])")
_COLORS = {"muted": 90, "stage": 94, "value": 96, "success": 92, "warning": 93, "error": 91}
_STATES = {
    "ERROR": "error",
    "NONFINITE": "error",
    "FAILED": "error",
    "ZERO": "warning",
    "WARNING": "warning",
    "OK": "success",
    "EXACT": "success",
    "SAVED": "success",
    "COMPLETE": "success",
    "SKIPPED_BUDGET": "muted",
    "UNAVAILABLE": "muted",
}


def color_enabled(stream: TextIO) -> bool:
    """Honor explicit color opt-out in terminals, notebooks and redirected output."""
    return stream.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"


def plain(text: object) -> str:
    """Strip terminal escapes and nonprinting controls from user-derived content."""
    return "".join(c for c in _ANSI.sub("", str(text)) if c in "\n\t" or unicodedata.category(c) != "Cc")


def paint(text: str, role: str, *, color: bool) -> str:
    """Keep color entirely within the presentation layer."""
    return f"\033[{_COLORS[role]}m{text}\033[0m" if color else text


def highlight(text: str, *, color: bool) -> str:
    """Emphasize numeric measurements without coloring digits inside parameter paths."""
    return _NUMBER.sub(lambda match: paint(match[0], "value", color=True), text) if color else text


def cell_width(text: str) -> int:
    """Count terminal cells, including wide CJK text and combining characters."""
    return sum(
        0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in plain(text)
    )


def clipped(text: str, width: int) -> str:
    """Truncate long identifiers visibly, without disturbing column alignment."""
    if cell_width(text) <= width:
        return text
    suffix = "." * min(3, width)
    result = ""
    for char in text:
        if cell_width(result + char) > width - len(suffix):
            break
        result += char
    return result + suffix


def wrapped(text: str, width: int) -> list[str]:
    """Wrap by display cells and preserve indentation in exception tracebacks."""
    text = plain(text).expandtabs(4)
    lines = []
    while cell_width(text) > width:
        cells = end = 0
        for end, char in enumerate(text):
            cells += cell_width(char)
            if cells > width and end > 0:
                break
        space = text.rfind(" ", 0, end + 1)
        cut = space if space > len(text) - len(text.lstrip()) else end
        lines.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    return [*lines, text]


def table_lines(
    columns: Sequence[str], rows: Sequence[Sequence[object]], *, width: int, right: Sequence[int] = (), color: bool
) -> list[str]:
    """Fit tabular results to the terminal; narrow displays use labeled row blocks."""
    headers = [plain(column).replace("\n", " ").expandtabs(1) for column in columns]
    cells = [[plain(value).replace("\n", " ").expandtabs(1) for value in row] for row in rows]
    widths = [max([cell_width(header), *(cell_width(row[i]) for row in cells)]) for i, header in enumerate(headers)]
    minimum = [min(value, max(6, cell_width(header))) for value, header in zip(widths, headers, strict=True)]
    available = width - 2 * (len(headers) - 1)
    if sum(minimum) > available:
        lines = []
        for row in cells:
            for header, value in zip(headers, row, strict=True):
                for line in wrapped(f"{header}: {value}", width):
                    lines.append(
                        paint(line, _STATES[value], color=color) if value in _STATES else highlight(line, color=color)
                    )
            lines.append("")
        return lines
    while sum(widths) > available:
        index = max(range(len(widths)), key=lambda i: widths[i] - minimum[i])
        widths[index] -= 1

    def render(row, *, header=False):
        rendered = []
        for index, (value, size) in enumerate(zip(row, widths, strict=True)):
            role = _STATES.get(value)
            value = clipped(value, size)
            padding = " " * (size - cell_width(value))
            cell = padding + value if index in right else value + padding
            if header:
                cell = paint(cell, "muted", color=color)
            elif role:
                cell = paint(cell, role, color=color)
            else:
                cell = highlight(cell, color=color)
            rendered.append(cell)
        return "  ".join(rendered).rstrip()

    rule = paint("-" * (sum(widths) + 2 * (len(widths) - 1)), "muted", color=color)
    return [render(headers, header=True), rule, *(render(row) for row in cells), rule]
