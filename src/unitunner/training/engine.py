"""Run sample-correct optimizer phases and shared training update scheduling."""

import math
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import closing, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from unitunner.artifacts.formats.precision import FloatingPrecision
from unitunner.checkpoints.saving import OptimizerSet, optimizer_roles
from unitunner.checkpoints.schema import ResumeIdentity
from unitunner.config.training import TrainingConfig
from unitunner.logging import set_context
from unitunner.precision import FLOAT_DTYPES
from unitunner.progress import status
from unitunner.training.checkpointing import save_training_update
from unitunner.training.contracts import RunState, StepResult
from unitunner.training.optimizers.gradients import normalize_gradients
from unitunner.training.optimizers.schedules import apply_learning_rate
from unitunner.training.progress import TrainingProgress
from unitunner.training.session import TrainingSession
from unitunner.training.telemetry import TrainingTelemetry
from unitunner.training.update_schedule import UpdateSchedule


@dataclass(frozen=True, slots=True)
class UpdateResult:
    """Record detached scalar mean loss and consumed samples for a completed update."""

    mean_loss: Tensor
    samples: int
    metrics: dict[str, Tensor] = field(default_factory=dict)
    grad_norm: Tensor | None = None
    microbatches: int = 1


def update_phase(
    steps: Iterable[StepResult],
    optimizer: Optimizer,
    state: RunState,
    *,
    phase: str = "main",
    max_grad_norm: float | None = None,
    precision: FloatingPrecision = "fp32",
    fp16_loss_scale: float = 128.0,
) -> UpdateResult:
    """Accumulate one phase and perform exactly one effective optimizer update.

    Losses must be per-sample means over this phase's trainable parameters.
    Weight each by sample count, backward immediately, then normalize gradients
    by total samples. Uneven final accumulation groups are supported. No batch
    metadata controls loop length. Nonfinite gradients fail before updating;
    state advances only after optimizer.step succeeds. BF16 forward autocast retains
    FP32 optimizer masters. FP16 uses explicit fixed loss scaling; nonfinite
    gradients fail before any optimizer step. Distributed optimizers are unsupported.
    """
    return update_phases(
        steps,
        {phase: optimizer},
        state,
        max_grad_norm=max_grad_norm,
        precision=precision,
        fp16_loss_scale=fp16_loss_scale,
    )[phase]


def update_phases(
    steps: Iterable[StepResult],
    optimizers: Mapping[str, Optimizer],
    state: RunState,
    *,
    max_grad_norm: float | None = None,
    precision: FloatingPrecision = "fp32",
    fp16_loss_scale: float = 128.0,
) -> dict[str, UpdateResult]:
    """Accumulate interleaved method phases and commit one completed data update.

    The method yields G and D graphs in its required forward order. Backward each
    immediately to avoid retaining generator graphs. Optimizers must own disjoint
    parameters on one device; each phase must consume the same sample/microbatch
    counts. Normalize each phase by its own samples, validate all losses/gradients,
    then step optimizers in mapping order. Count each data microbatch only once.

    A failed optimizer.step may leave earlier optimizers updated: discard that
    failed session rather than checkpointing it. No automatic numerical rollback,
    distributed reduction or shared-parameter phase semantics are implied.
    """
    if not optimizers or any(not name for name in optimizers):
        raise ValueError("Optimizer phases must be nonempty and named")
    parameters = {
        name: [parameter for group in optimizer.param_groups for parameter in group["params"]]
        for name, optimizer in optimizers.items()
    }
    all_parameters = [parameter for group in parameters.values() for parameter in group]
    if any(not group for group in parameters.values()) or len({id(p) for p in all_parameters}) != len(all_parameters):
        raise ValueError("Optimizer parameters must be nonempty and unique across phases")
    if any(parameter.device != all_parameters[0].device for parameter in all_parameters):
        raise ValueError("A local optimizer cycle requires one device")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm < 0):
        raise ValueError("max_grad_norm must be finite and nonnegative")
    if not math.isfinite(fp16_loss_scale) or fp16_loss_scale < 1:
        raise ValueError("FP16 loss scale must be finite and at least one")
    loss_scale = fp16_loss_scale if precision == "fp16" else 1.0
    totals: dict[str, Tensor] = {}
    metrics: dict[str, dict[str, Tensor]] = {name: {} for name in optimizers}
    grad_norms: dict[str, Tensor] = {}
    samples = dict.fromkeys(optimizers, 0)
    microbatches = dict.fromkeys(optimizers, 0)
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    try:
        for result in _forward_steps(steps, all_parameters[0].device, precision):
            phase = result.phase
            set_context(
                update=state.updates + 1, phase=phase, microbatch=microbatches.get(phase, 0) + 1, stage="backward"
            )
            if phase not in optimizers:
                raise ValueError(f"Unexpected optimizer phase {phase!r}")
            if microbatches[phase] and result.metrics.keys() != metrics[phase].keys():
                raise ValueError("Step metric keys must be consistent within an optimizer phase")
            # Promote before weighting/summing, retaining FP64 reference math.
            loss = result.loss.float() if result.loss.dtype in (torch.float16, torch.bfloat16) else result.loss
            weighted_loss = loss * result.sample_count
            (weighted_loss * loss_scale).backward() if loss_scale != 1 else weighted_loss.backward()
            detached = weighted_loss.detach()
            totals[phase] = detached if phase not in totals else totals[phase] + detached
            for name, value in result.metrics.items():
                if value.ndim != 0:
                    raise ValueError("Step metrics must be scalar means")
                detached_metric = value.detach()
                if detached_metric.dtype in (torch.float16, torch.bfloat16):
                    detached_metric = detached_metric.float()
                weighted = detached_metric * result.sample_count
                metrics[phase][name] = metrics[phase].get(name, torch.zeros_like(weighted)) + weighted
            samples[phase] += result.sample_count
            microbatches[phase] += 1
        if totals.keys() != optimizers.keys():
            raise ValueError("An update requires at least one microbatch in every phase")
        if len(set(samples.values())) != 1 or len(set(microbatches.values())) != 1:
            raise ValueError("All optimizer phases must consume the same data samples and microbatches")
        for name, group in parameters.items():
            set_context(phase=name, stage="gradient validation and clipping")
            if not torch.isfinite(totals[name]):
                raise FloatingPointError("Optimizer phase has a non-finite loss")
            if not any(parameter.grad is not None for parameter in group):
                raise ValueError("Phase loss produced no optimizer gradients")
            if loss_scale != 1:
                for parameter in group:
                    if parameter.grad is not None:
                        parameter.grad.div_(loss_scale)
            normalize_gradients([parameter.grad for parameter in group if parameter.grad is not None], samples[name])
            grad_norms[name] = nn.utils.clip_grad_norm_(
                group, float("inf") if max_grad_norm is None else max_grad_norm, error_if_nonfinite=True
            )
        for name, optimizer in optimizers.items():
            set_context(phase=name, stage="optimizer update")
            optimizer.step()
    except BaseException:
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        raise
    state.updates += 1
    state.microbatches += next(iter(microbatches.values()))
    state.samples += next(iter(samples.values()))
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    return {
        name: UpdateResult(
            mean_loss=totals[name] / samples[name],
            samples=samples[name],
            metrics={key: value / samples[name] for key, value in metrics[name].items()},
            grad_norm=grad_norms[name].detach(),
            microbatches=microbatches[name],
        )
        for name in optimizers
    }


def run_updates(
    config: TrainingConfig,
    steps: Callable[[], Iterable[StepResult]],
    *,
    optimizer: OptimizerSet,
    modules: Mapping[str, nn.Module],
    state: RunState,
    identity: ResumeIdentity,
    device: torch.device,
    weights_writer: Callable[[int], object],
    preview_writer: Callable[[int], None] | None = None,
) -> RunState:
    """Advance a restored local run without owning model loading or noise sampling.

    Each steps() invocation produces one accumulation group. The shared engine
    normalizes by consumed samples; scheduling uses completed update counts.
    Save complete optimizer/module/RNG state only after a successful update and
    before preview. Callers own preview RNG/mode restoration and all resources.
    """
    precision = config.execution.precision
    if precision == "auto":
        raise ValueError("Training precision must be resolved before running updates")
    output = Path(config.output.directory)
    output.mkdir(parents=True, exist_ok=True)
    schedule = UpdateSchedule.from_config(config)
    optimizers = optimizer_roles(optimizer)
    with TrainingSession() as session:
        telemetry = session.own(
            TrainingTelemetry(
                config.output.tensorboard, state, name=config.output.name, run_name=config.output._run_name
            )
        )
        if schedule.initial_preview_due(state.updates) and preview_writer is not None:
            status("[train] Rendering initial preview")
            preview_writer(0)
        status(f"[train] Ready; accumulation={config.train.accumulation_steps}, max_updates={config.train.max_updates}")
        progress = session.own(
            TrainingProgress(config.train.max_updates, initial=state.updates, interval=config.output.log_every_seconds)
        )
        while state.updates < config.train.max_updates:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = monotonic()
            apply_learning_rate(config.train, state.updates, optimizers.values())
            group = steps()
            with closing(group) if isinstance(group, Generator) else nullcontext(group) as update_steps:
                results = update_phases(
                    update_steps,
                    optimizers,
                    state,
                    max_grad_norm=config.train.max_grad_norm,
                    precision=precision,
                    fp16_loss_scale=config.execution.fp16_loss_scale,
                )
            scalars = telemetry.record(results, optimizers, state, monotonic() - started, device)
            progress.report(state.updates, state.samples, scalars)
            if schedule.save_due(state.updates):
                status(f"[train] Saving checkpoint at update {state.updates}")
                save_training_update(
                    config,
                    modules=modules,
                    optimizer=optimizer,
                    state=state,
                    identity=identity,
                    device=device,
                    weights_writer=weights_writer,
                )
            if schedule.preview_due(state.updates) and preview_writer is not None:
                status(f"[train] Rendering preview at update {state.updates}")
                preview_writer(state.updates)
    return state


def _forward_steps(
    steps: Iterable[StepResult], device: torch.device, precision: FloatingPrecision
) -> Iterator[StepResult]:
    """Autocast only forward graph construction; backward/optimizer/previews run outside.

    Disable weight-cast caching because VBAR rebinds frozen parameter storage.
    Nonreentrant checkpoint captures this autocast policy for backward replay.
    BF16 uses FP32 optimizer masters and does not require FP16 loss scaling.
    """
    if precision == "fp32":
        yield from steps
        return
    if precision not in ("bf16", "fp16"):
        raise ValueError("Training compute precision must be fp32, fp16 or bf16")
    iterator = iter(steps)
    while True:
        try:
            with torch.autocast(device.type, dtype=FLOAT_DTYPES[precision], cache_enabled=False):
                result = next(iterator)
        except StopIteration:
            return
        yield result
