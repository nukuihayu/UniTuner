"""Same-directory atomic publication for local weight and cache files."""

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_output(destination: str | Path) -> Generator[Path]:
    """Yield a temporary path and publish it only after successful durable writing.

    The temporary file shares the destination filesystem. On success its data
    is fsynced before os.replace; on failure the old destination remains intact
    and the temporary file is removed. Writer coordination is owned by callers.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
