"""Write completed optimizer updates to optional TensorBoard event files."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.optim import Optimizer

from unitunner.training.output import create_run_directory
from unitunner.training.state import RunState

if TYPE_CHECKING:
    from unitunner.training.engine import UpdateResult


class TrainingTelemetry:
    """Own an optional writer and flush it on normal or exceptional session exit."""

    def __init__(self, directory: str | None, state: RunState, *, name: str, run_name: str | None = None) -> None:
        self.writer = None
        self.directory: Path | None = None
        if directory is not None:
            from torch.utils.tensorboard import SummaryWriter

            root = Path(directory)
            if run_name is not None:
                self.directory = root / run_name
                self.directory.mkdir(parents=True, exist_ok=True)
            else:
                self.directory = create_run_directory(root, name)
            self.writer = SummaryWriter(
                log_dir=str(self.directory), purge_step=state.updates + 1 if state.updates else None, flush_secs=10
            )

    def record(
        self,
        results: Mapping[str, UpdateResult],
        optimizers: Mapping[str, Optimizer],
        state: RunState,
        seconds: float,
        device: torch.device,
    ) -> dict[str, float]:
        """Return one scalar snapshot for console logs and optional TensorBoard events."""
        values = {"train/samples": state.samples, "train/microbatches": state.microbatches}
        scalars: dict[str, float] = {key: float(value) for key, value in values.items()}
        first = next(iter(results.values()))
        scalars["train/update_samples"] = float(first.samples)
        scalars["train/update_microbatches"] = float(first.microbatches)
        scalars["performance/update_seconds"] = seconds
        scalars["performance/samples_per_second"] = first.samples / max(seconds, 1e-9)
        for phase, result in results.items():
            scalars[f"loss/{phase}"] = float(result.mean_loss.cpu())
            for name, value in result.metrics.items():
                scalars[f"loss/{phase}/{name}"] = float(value.cpu())
            if result.grad_norm is not None:
                scalars[f"train/{phase}/grad_norm"] = float(result.grad_norm.cpu())
            for index, group in enumerate(optimizers[phase].param_groups):
                scalars[f"train/{phase}/lr/{index}"] = float(group["lr"])
        if device.type == "cuda":
            scalars["memory/allocated_gib"] = torch.cuda.memory_allocated(device) / 2**30
            scalars["memory/reserved_gib"] = torch.cuda.memory_reserved(device) / 2**30
            scalars["memory/peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        if self.writer is not None:
            for name, value in scalars.items():
                self.writer.add_scalar(name, value, state.updates)
        return scalars

    def close(self) -> None:
        """Flush buffered events and release the writer once."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
