"""Hash content without constructing whole-file byte buffers."""

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import JsonValue


def file_sha256(path: Path, *, release_cache: bool = False) -> str:
    """Hash content, optionally asking the OS to release scanned file pages.

    Dataset preflight reads files once and can otherwise fill Linux container
    RAM with page cache. Advice is best effort and never changes validation.
    """
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        advise = getattr(os, "posix_fadvise", None)
        dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
        if release_cache and advise is not None and dontneed is not None:
            try:
                advise(stream.fileno(), 0, 0, dontneed)
            except OSError:
                # Some filesystems do not implement this optional hint.
                pass
        return digest


def canonical_sha256(value: JsonValue) -> str:
    """Hash sorted UTF-8 JSON, rejecting nonfinite numbers and non-JSON values."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonical_sequence_sha256(items: Iterable[JsonValue], *, field: str, context: Mapping[str, JsonValue]) -> str:
    """Hash a canonical JSON object with one streamed array, buffering only one item.

    Produce exactly canonical_sha256({**context, field: list(items)}) without
    retaining the iterable or constructing its complete JSON string.
    """
    if field in context:
        raise ValueError("Streamed array field must not occur in context")

    def encode(value: JsonValue) -> bytes:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )

    digest = hashlib.sha256(b"{")
    for index, key in enumerate(sorted((*context, field))):
        if index:
            digest.update(b",")
        digest.update(encode(key) + b":")
        if key != field:
            digest.update(encode(context[key]))
            continue
        digest.update(b"[")
        for offset, item in enumerate(items):
            if offset:
                digest.update(b",")
            digest.update(encode(item))
        digest.update(b"]")
    digest.update(b"}")
    return digest.hexdigest()


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
