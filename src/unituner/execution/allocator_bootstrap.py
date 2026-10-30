"""Initialize optional AIMDO hooks before Torch, without loading models or choosing memory budgets."""

import sys
from importlib import import_module
from importlib.metadata import version
from importlib.util import find_spec
from typing import Protocol, cast


class _Control(Protocol):
    def init(self) -> bool: ...


_initialized = False


def initialize_vbar() -> None:
    """Load pinned AIMDO hooks before Torch import; CLI owns this process-wide opt-in.

    Device contexts are initialized later after Torch enumerates visible devices.
    Python applications must call this function before importing Torch as well.
    """
    global _initialized
    if _initialized:
        return
    if "torch" in sys.modules:
        raise ValueError("VBAR must be initialized before importing Torch; use the unituner CLI or initialize_vbar()")
    if find_spec("comfy_aimdo") is None:
        raise ValueError("VBAR requires the optional backend: uv sync --extra vbar")
    if version("comfy-aimdo") != "0.6.0":
        raise ValueError("VBAR requires comfy-aimdo==0.6.0")
    control = cast(_Control, import_module("comfy_aimdo.control"))
    if not control.init():
        raise ValueError("AIMDO hooks could not initialize on this platform")
    _initialized = True


def vbar_initialized() -> bool:
    """Report successful process bootstrap without importing the optional backend."""
    return _initialized
