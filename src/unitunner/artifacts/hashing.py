"""Hash content without constructing whole-file byte buffers."""

import hashlib
import json
from pathlib import Path

from pydantic import JsonValue


def file_sha256(path: Path) -> str:
    """Return a bounded-memory content digest; mtime is not a content identity."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value: JsonValue) -> str:
    """Hash sorted UTF-8 JSON, rejecting nonfinite numbers and non-JSON values."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def directory_sha256(directory: Path) -> str:
    """Hash visible artifact files and relative paths without buffering weights.

    Exclude hidden cache/git directories. All visible regular files participate,
    including tokenizer templates/config and sharded weights; an empty tree fails.
    """
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )
    if not files:
        raise ValueError("Artifact directory contains no files")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0" + file_sha256(path).encode() + b"\n")
    return digest.hexdigest()
