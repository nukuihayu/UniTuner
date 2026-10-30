"""Resolve one execution identity for artifacts, diagnostic logs and TensorBoard."""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from unituner.artifacts.atomic import atomic_output
from unituner.config.training import TrainingConfig


def _safe_run_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("._-") or "model"


def create_run_directory(root: Path, name: str) -> Path:
    """Allocate a named timestamp directory, retrying atomic mkdir on collisions."""
    base_name = f"{_safe_run_name(name)}_{datetime.now():%Y%m%d-%H%M%S-%f}"
    suffix = 0
    while True:
        directory = root / (base_name if suffix == 0 else f"{base_name}-{suffix}")
        try:
            directory.mkdir(parents=True)
        except FileExistsError:
            suffix += 1
            continue
        return directory


def weights_path(config: TrainingConfig, *, update: int | None = None, encoder: bool = False) -> Path:
    """Locate consumer weights independently of exact training continuation files."""
    suffix = "" if update is None else f"-{update:08d}"
    component = ".encoder" if encoder else ""
    return Path(config.output.directory) / "weights" / f"{config.output.name}{suffix}{component}.safetensors"


def resume_path(config: TrainingConfig, update: int) -> Path:
    """Locate a full training-state snapshot intended only for --resume."""
    return Path(config.output.directory) / "resume" / f"{config.output.name}.state-{update:08d}.safetensors"


def publish_latest_weights(config: TrainingConfig, update: int) -> None:
    """Update latest inference files at each save boundary without rewriting tensor data."""
    for encoder in (False, True):
        source = weights_path(config, update=update, encoder=encoder)
        if encoder and not source.is_file():
            continue
        with atomic_output(weights_path(config, encoder=encoder)) as temporary:
            temporary.unlink()
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copyfile(source, temporary)


def prune_checkpoints(config: TrainingConfig) -> None:
    """Apply independent retention only after the new weight/state pair was saved."""
    root = Path(config.output.directory)
    name = re.escape(config.output.name)
    for section, limit, pattern in (
        ("resume", config.output.max_resume_checkpoints, rf"{name}\.state-(\d{{8,}})\.safetensors"),
        ("weights", config.output.max_weight_checkpoints, rf"{name}-(\d{{8,}})\.safetensors"),
    ):
        directory = root / section
        if limit == 0 or not directory.is_dir() or directory.is_symlink():
            continue
        snapshots = []
        for path in directory.iterdir():
            match = re.fullmatch(pattern, path.name)
            if match is not None and path.is_file():
                snapshots.append((int(match[1]), path))
        for _, path in sorted(snapshots, key=lambda item: item[0], reverse=True)[limit:]:
            path.unlink()
            if section == "weights":
                path.with_suffix(".encoder.safetensors").unlink(missing_ok=True)


def prepare_training_output(config: TrainingConfig, *, resume: Path | None = None) -> TrainingConfig:
    """Resolve output once; fresh fixed runs replace only recognized training files."""
    if config.output._run_name is not None:
        return config
    fixed = "directory" in config.output.model_fields_set
    if fixed:
        run_name = _safe_run_name(config.output.name)
        directory = Path(config.output.directory)
        directory.mkdir(parents=True, exist_ok=True)
    else:
        directory = create_run_directory(Path("runs"), config.output.name)
        run_name = directory.name
    if fixed and resume is None:
        # Match literal filenames, never interpret the user-supplied name as a glob.
        candidates = list(directory.iterdir())
        for section in ("weights", "resume"):
            child = directory / section
            if child.is_dir() and not child.is_symlink():
                candidates.extend(child.iterdir())
        for path in candidates:
            name = path.name
            if path.is_file() and (
                name in {f"{config.output.name}.safetensors", f"{config.output.name}.encoder.safetensors"}
                or (
                    name.startswith(f"{config.output.name}.state-")
                    and re.fullmatch(r"\d{8,}\.safetensors", name[len(config.output.name) + 7 :])
                )
                or (
                    path.parent == directory / "weights"
                    and name.startswith(f"{config.output.name}-")
                    and re.fullmatch(r"\d{8,}(?:\.encoder)?\.safetensors", name[len(config.output.name) + 1 :])
                )
            ):
                path.unlink()
        previews = directory / "previews"
        if previews.is_dir() and not previews.is_symlink():
            for path in previews.iterdir():
                if path.is_file() and re.fullmatch(r"step-\d{8,}-\d{2,}\.(png|mp4|wav)", path.name):
                    path.unlink()
        if config.output.tensorboard is not None:
            events = Path(config.output.tensorboard) / run_name
            if events.is_dir() and not events.is_symlink():
                for path in events.glob("events.out.tfevents.*"):
                    if path.is_file():
                        path.unlink()
    output = config.output.model_copy(update={"directory": str(directory)})
    output._run_name = run_name
    return config.model_copy(update={"output": output})
