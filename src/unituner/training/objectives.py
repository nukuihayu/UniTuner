"""Own scoped LayerSync target features without persistent model graphs or global hooks."""

from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from typing import Protocol

import torch
from torch import Tensor

from unituner.config.features import LayerSyncConfig, RepaConfig
from unituner.config.restoration import MethodConfig
from unituner.config.routing import TreadConfig
from unituner.data.cache.representation import RepaTargets
from unituner.data.cache.storage import CacheEntry
from unituner.execution.regions import ModelRegions
from unituner.models.components.features import FeatureRecorder
from unituner.training.contracts import StepResult
from unituner.training.losses.representation import RepaProjection, VisualGrid, layersync_loss
from unituner.training.session import TrainingSession


class FeatureModel(Protocol):
    """Expose native target capture separately from model state dictionaries."""

    feature_recorder: FeatureRecorder | None
    feature_width: int
    feature_patch: tuple[int, int, int]


class FeatureAlignment:
    """Borrow a native model and own only one active student's selected feature graphs."""

    def __init__(
        self,
        model: FeatureModel,
        config: LayerSyncConfig | None,
        *,
        num_layers: int,
        repa: RepaConfig | None = None,
        targets: RepaTargets | None = None,
        device: torch.device | None = None,
    ) -> None:
        selected = set() if config is None else {config.student_layer, config.teacher_layer}
        if repa is not None:
            selected.add(repa.layer)
        if not selected or max(selected) >= num_layers:
            raise ValueError(f"Feature objectives require selected layers within {num_layers} native visual blocks")
        if model.feature_recorder is not None:
            raise ValueError("Native model already has a feature capture owner")
        if (repa is None) != (targets is None) or (repa is not None and device is None):
            raise ValueError("REPA requires verified targets and an explicit projector device")
        self.model: FeatureModel = model
        self.config: LayerSyncConfig | None = config
        self.repa: RepaConfig | None = repa
        self.targets: RepaTargets | None = targets
        self.projection: RepaProjection | None = (
            None if repa is None else RepaProjection(model.feature_width, repa.encoder_dim).to(device=device)
        )
        self.recorder: FeatureRecorder = FeatureRecorder(frozenset(selected))
        self._target_rows: list[tuple[Tensor, VisualGrid]] = []
        self._closed: bool = False
        model.feature_recorder = self.recorder

    def set_targets(self, entries: tuple[CacheEntry, ...], latents: Tensor, *, skip_first_frame: bool = False) -> None:
        """Bind identity-checked frozen encoder grids to this microbatch's native visual geometry."""
        if self.targets is None:
            return
        if (
            not self.recorder.active
            or self._target_rows
            or latents.ndim not in (4, 5)
            or len(entries) != latents.shape[0]
        ):
            raise ValueError("REPA targets require one active microbatch with BCHW/BCFHW native latents")
        frames, height, width = (1 if latents.ndim == 4 else latents.shape[2]), latents.shape[-2], latents.shape[-1]
        if skip_first_frame:
            if latents.ndim != 5 or frames < 2:
                raise ValueError("REPA skipping the base requires multiple independent latent layers")
            frames -= 1
        temporal, vertical, horizontal = self.model.feature_patch
        if frames % temporal or height % vertical or width % horizontal:
            raise ValueError("REPA native latent grid is not divisible by the model visual patch")
        grid = VisualGrid(frames // temporal, height // vertical, width // horizontal)
        rows = [self.targets.read(entry) for entry in entries]
        if skip_first_frame:
            if any(row.shape[0] < 2 for row in rows):
                raise ValueError("REPA skipping the base requires matching composite/foreground encoder frames")
            rows = [row[1:] for row in rows]
        self._target_rows = [(row, grid) for row in rows]

    @contextmanager
    def capture(self) -> Generator[None]:
        """Collect only this forward group, releasing graphs even when the model or loss fails."""
        if self._closed or self.recorder.active:
            raise RuntimeError("LayerSync capture is closed or already active")
        self.recorder.active = True
        try:
            yield
        finally:
            self.recorder.clear()
            self._target_rows.clear()

    def step(self, result: StepResult) -> StepResult:
        """Add original stopped-teacher cosine loss to the native differentiable mean."""
        if not self.recorder.active:
            raise RuntimeError("Feature loss requires active native target capture")
        loss = result.loss
        metrics = dict(result.metrics)
        if self.config is not None:
            student = self.recorder.features.get(self.config.student_layer, [])
            teacher = self.recorder.features.get(self.config.teacher_layer, [])
            if not student or len(student) != len(teacher):
                raise RuntimeError("LayerSync requires matching native target captures")
            count = sum(value.shape[0] for value in student)
            alignment = sum(
                (
                    layersync_loss(left, right) * left.shape[0] / count
                    for left, right in zip(student, teacher, strict=True)
                ),
                start=loss.new_zeros(()),
            )
            loss = loss + self.config.weight * alignment
            metrics["layersync"] = alignment.detach()
        if self.repa is not None and self.projection is not None:
            captures = self.recorder.features.get(self.repa.layer, [])
            rows = [row.unsqueeze(0) for capture in captures for row in capture.unbind()]
            if not rows or len(rows) != len(self._target_rows):
                raise ValueError("REPA captures do not match the ordered native target cache records")
            alignment = sum(
                (
                    self.projection.alignment_loss(row, target.unsqueeze(0), grid)
                    for row, (target, grid) in zip(rows, self._target_rows, strict=True)
                ),
                start=loss.new_zeros(()),
            ) / len(rows)
            loss = loss + self.repa.weight * alignment
            metrics["repa"] = alignment.detach()
        return StepResult(loss, result.sample_count, result.phase, metrics)

    def close(self) -> None:
        """Release capture and detach only this owner's recorder, idempotently."""
        if not self._closed:
            if self.model.feature_recorder is not self.recorder:
                raise RuntimeError("LayerSync must close after later feature owners")
            self.recorder.clear()
            self._target_rows.clear()
            self.model.feature_recorder = None
            self._closed = True


def align_steps(steps: Callable[[], Iterator[StepResult]], alignment: FeatureAlignment | None) -> Iterator[StepResult]:
    """Finish and clear one native feature scope before yielding its graph to backward."""
    if alignment is None:
        yield from steps()
        return
    iterator = steps()
    while True:
        with alignment.capture():
            result = next(iterator, None)
            if result is None:
                return
            aligned = alignment.step(result)
        yield aligned


class RoutedModel(Protocol):
    """Expose an optional native target routing policy, excluded from state dictionaries."""

    token_routing: TreadConfig | None


class TokenRoutingInstallation:
    """Borrow a native model and restore its routing policy after outstanding backward work."""

    def __init__(self, model: RoutedModel, config: TreadConfig, *, num_layers: int) -> None:
        if config.end_layer > num_layers:
            raise ValueError(f"TREAD end_layer exceeds {num_layers} native visual blocks")
        if model.token_routing is not None:
            raise ValueError("Native model already has a token routing owner")
        self.model: RoutedModel = model
        self.config: TreadConfig = config
        self._closed: bool = False
        model.token_routing = config

    def close(self) -> None:
        """Detach only this owner's immutable policy, idempotently."""
        if not self._closed:
            if self.model.token_routing is not self.config:
                raise RuntimeError("Token routing must close after later owners")
            self.model.token_routing = None
            self._closed = True


class ObjectiveModel(FeatureModel, RoutedModel, Protocol):
    """Expose feature capture and token routing on a native training model."""


def install_objectives(
    model: ObjectiveModel,
    regions: ModelRegions,
    method: MethodConfig,
    *,
    session: TrainingSession,
    targets: RepaTargets | None,
    device: torch.device,
) -> FeatureAlignment | None:
    """Install routing then feature capture, registering each owner before the next.

    The caller's session rolls back setup failures and releases objectives after
    execution owners. Feature projection construction retains its original RNG order.
    """
    num_layers = sum(region.branch in ("visual", "joint") for region in regions.regions)
    if method.tread is not None:
        session.own(TokenRoutingInstallation(model, method.tread, num_layers=num_layers))
    if method.layersync is None and method.repa is None:
        return None
    return session.own(
        FeatureAlignment(
            model, method.layersync, num_layers=num_layers, repa=method.repa, targets=targets, device=device
        )
    )
