"""Publish inference weights and resume state in one shared update-boundary order."""

from collections.abc import Callable, Mapping
from dataclasses import asdict

import torch
from torch import nn

from unituner.checkpoints.saving import OptimizerSet, save_checkpoint
from unituner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unituner.config.training import TrainingConfig
from unituner.training.contracts import RunState
from unituner.training.output import prune_checkpoints, resume_path


def save_training_update(
    config: TrainingConfig,
    *,
    modules: Mapping[str, nn.Module],
    optimizer: OptimizerSet,
    state: RunState,
    identity: ResumeIdentity,
    device: torch.device,
    weights_writer: Callable[[int], object],
) -> None:
    """Write weights, snapshot continuation state, then prune completed snapshots.

    Call only after a successful update, before preview. Preserve a single
    optimizer's main role and restoration's explicit generator/discriminator roles.
    A failed publication or snapshot never triggers retention pruning.
    """
    weights_writer(state.updates)
    save_checkpoint(
        resume_path(config, state.updates),
        modules,
        optimizer,
        CheckpointCursor(**asdict(state)),
        identity,
        cuda_devices=(device.index or 0,) if device.type == "cuda" else (),
        reference_frozen_parameters=config.output.checkpoint_storage == "frozen-reference",
    )
    prune_checkpoints(config)
