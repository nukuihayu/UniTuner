"""Evaluate deterministic learning rates from the completed effective-update cursor."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

from unituner.config.train import TrainConfig

if TYPE_CHECKING:
    from torch.optim import Optimizer


def apply_learning_rate(config: TrainConfig, completed_updates: int, optimizers: Iterable[Optimizer]) -> float:
    """Apply the next-update base rate while preserving each group multiplier."""
    rate = learning_rate(config, completed_updates)
    groups = [group for optimizer in optimizers for group in optimizer.param_groups]
    scales = [group.get("learning_rate_scale", 1.0) for group in groups]
    if any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("Optimizer learning-rate scales must be finite and positive")
    for group, scale in zip(groups, scales, strict=True):
        group["lr"] = rate * scale
    return rate


def learning_rate(config: TrainConfig, completed_updates: int) -> float:
    """Return the LR for the next update using native constant/linear/cosine formulas.

    Warmup factor is step/warmup. Afterwards linear uses 1-progress and cosine
    uses (1+cos(pi*progress))/2, where progress=(step-warmup)/(horizon-warmup).
    This matches the corresponding Hugging Face warmup schedules at cycle=0.5.
    Constant with zero warmup preserves the original unscheduled LR. No mutable
    scheduler state exists: config identity and the checkpoint cursor determine
    every rate, independent of accumulation, logging and save/preview frequency.
    """
    if not 0 <= completed_updates <= config.max_updates:
        raise ValueError("Learning-rate update cursor must be inside the configured horizon")
    if completed_updates < config.warmup_updates:
        factor = completed_updates / config.warmup_updates
    elif config.lr_schedule == "constant":
        factor = 1.0
    else:
        progress = (completed_updates - config.warmup_updates) / (config.max_updates - config.warmup_updates)
        factor = 1 - progress if config.lr_schedule == "linear" else (1 + math.cos(math.pi * progress)) / 2
    return config.learning_rate * factor
