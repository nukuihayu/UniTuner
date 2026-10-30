"""Stage fresh cache directories and roll back publication failures."""

import re
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

from unituner.config.cache import CacheConfig
from unituner.progress import status

_RECORD = re.compile(r"[0-9a-f]{64}\.safetensors")


@contextmanager
def fresh_cache_config(config: CacheConfig) -> Iterator[CacheConfig]:
    """Redirect every dataset and optional REPA output to a fresh staging directory."""
    destinations = [Path(group.cache_directory).resolve() for group in config.dataset.groups.values()]
    if config.repa is not None:
        destinations.append(Path(config.repa.cache_directory).resolve())
    protected = [
        Path(value).resolve() for value in (config.model.vae, config.model.text_encoder, config.model.assets) if value
    ]
    for group in config.dataset.groups.values():
        for field in ("source", "low_quality_source", "mask_source", "control_source"):
            value = getattr(group, field, None)
            if value:
                protected.append(Path(value).resolve())
        for field in ("reference_sources", "layer_sources"):
            protected.extend(Path(value).resolve() for value in getattr(group, field, ()))
    for destination in destinations:
        if any(path.is_relative_to(destination) for path in protected):
            raise ValueError(f"Cache destination must not contain input data or model assets: {destination}")
    with rebuild_directories(destinations) as staged:
        groups = {
            name: group.model_copy(update={"cache_directory": str(staged[Path(group.cache_directory).resolve()])})
            for name, group in config.dataset.groups.items()
        }
        repa = config.repa
        if repa is not None:
            repa = repa.model_copy(update={"cache_directory": str(staged[Path(repa.cache_directory).resolve()])})
        yield config.model_copy(update={"dataset": config.dataset.model_copy(update={"groups": groups}), "repa": repa})


def _preserve_unmanaged(previous: Path, fresh: Path) -> None:
    for path in previous.iterdir():
        if path.name == "manifest.json" or _RECORD.fullmatch(path.name):
            continue
        target = fresh / path.name
        if target.exists():
            raise ValueError(f"New cache conflicts with an unrelated existing file: {path}")
        if path.is_dir() and not path.is_symlink():
            shutil.copytree(path, target, symlinks=True)
        else:
            shutil.copy2(path, target, follow_symlinks=False)


def _cleanup(root: Path, publication: dict[str, bool]) -> None:
    """Never remove the only surviving backup if filesystem rollback itself fails."""
    if not publication["complete"] and (root / "previous").exists():
        status(f"[cache] Rollback needs attention; original cache preserved at {root / 'previous'}", level="ERROR")
        return
    shutil.rmtree(root)


@contextmanager
def rebuild_directories(destinations: Sequence[Path]) -> Iterator[dict[Path, Path]]:
    """Build without reuse, then replace all outputs with rollback on rename errors.

    Preserve unrelated files; obsolete content-addressed records are removed.
    Temporary directories share each destination's filesystem. Concurrent readers
    and writers must be stopped while publishing the directory replacements.
    """
    paths = [path.resolve() for path in destinations]
    for index, path in enumerate(paths):
        if path == Path(path.anchor) or (path.exists() and not path.is_dir()):
            raise ValueError(f"Invalid cache destination: {path}")
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in paths[:index]):
            raise ValueError("Cache destinations must be distinct and must not contain one another")
    with ExitStack() as stack:
        publication = {"complete": False}
        staged: dict[Path, Path] = {}
        roots: dict[Path, Path] = {}
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            root = Path(tempfile.mkdtemp(prefix=f".{path.name}-rebuild-", dir=path.parent))
            stack.callback(_cleanup, root, publication)
            roots[path] = root
            staged[path] = root / "fresh"
            staged[path].mkdir()
            status(f"[cache] Rebuild {path} (old cache kept until success)")
        yield staged
        for path, fresh in staged.items():
            if not (fresh / "manifest.json").is_file():
                raise ValueError(f"Rebuilt cache has no manifest: {path}")
            if path.exists():
                _preserve_unmanaged(path, fresh)
        moved: list[Path] = []
        published: list[Path] = []
        try:
            for path, fresh in staged.items():
                if path.exists():
                    path.rename(roots[path] / "previous")
                    moved.append(path)
                fresh.rename(path)
                published.append(path)
        except BaseException:
            for path in reversed(published):
                try:
                    path.rename(staged[path])
                except OSError as error:
                    status(f"[cache] Cannot withdraw new cache {path}: {error}", level="ERROR")
            for path in reversed(moved):
                try:
                    (roots[path] / "previous").rename(path)
                except OSError as error:
                    status(f"[cache] Cannot restore original cache {path}: {error}", level="ERROR")
            raise
        publication["complete"] = True
        for path in paths:
            status(f"[cache] Published fresh cache: {path}")
