"""Assemble cached native H3 full/adapter training with shared local update and checkpoint owners."""

from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from unituner.analysis.context import publication_metadata, record_context
from unituner.artifacts.hashing import canonical_sha256, file_sha256
from unituner.artifacts.tensors import weight_artifact_sha256
from unituner.checkpoints.restore import restore_checkpoint
from unituner.checkpoints.schema import build_resume_identity
from unituner.config.training import TrainingConfig
from unituner.data.cache.representation import RepaTargets, representation_dataset_sha256
from unituner.data.datasets.batching import DatasetBatches
from unituner.execution.frozen_blocks import FrozenModelPlacement
from unituner.execution.installation import (
    ExecutionInstallation,
    prepare_base_weights,
    training_base_dtype,
    validate_training_execution,
)
from unituner.execution.precision import resolve_training_precision
from unituner.execution.runtime import runtime_scope
from unituner.models.catalog import ModelFamily
from unituner.models.minimax_h3.config.training import MiniMaxH3FlowConfig
from unituner.models.minimax_h3.data.batching import build_minimax_h3_batches
from unituner.models.minimax_h3.data.cache.storage import (
    MiniMaxH3CacheFormat,
    load_minimax_h3_manifest,
    read_minimax_h3_record,
)
from unituner.models.minimax_h3.loading import load_minimax_h3, save_minimax_h3
from unituner.precision import FLOAT_DTYPES
from unituner.training.contracts import RunState, StepResult
from unituner.training.engine import run_updates
from unituner.training.integrations.minimax_h3.binding import MiniMaxH3Binding
from unituner.training.integrations.minimax_h3.inputs import sample_minimax_h3_request
from unituner.training.objectives import align_steps, install_objectives
from unituner.training.optimizers.adamw import create_adamw
from unituner.training.output import publish_latest_weights, weights_path
from unituner.training.parameters import ParameterGroup, ParameterInstallation
from unituner.training.publishing import publish_adapter
from unituner.training.session import TrainingSession

if TYPE_CHECKING:
    from unituner.training.integrations.minimax_h3.previews import AudioVideoPreviews


@runtime_scope
def train_minimax_h3(config: TrainingConfig, *, device: torch.device, resume: Path | None = None) -> RunState:
    """Train native FP32 AV full weights or adapters from complete verified single-record caches.

    Keep original noise ordering and the trainer time roundtrip. Audio presence
    controls supervision, including single-frame silent targets. Restore full
    optimizer/model/adapter/RNG/cursor state before drawing the next sample.
    Export algorithm-specific native names with alpha; explicit consumer export
    handles ComfyUI's independent namespace. Optional previews own their VAEs
    and condition snapshots without changing the numerical training cursor.
    """
    config = resolve_training_precision(config)
    adapter, execution = config.train.adapter, config.execution
    if (
        config.model.family != ModelFamily.MINIMAX_H3
        or config.method.kind != "flow-matching"
        or any(data.kind not in ("image", "video") for data in config.dataset.groups.values())
    ):
        raise ValueError("MiniMaxH3 training requires cached image/video flow in full or adapter mode")
    validate_training_execution(execution, device=device, adapter=config.train.adapter is not None)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Training execution requires CPU/CUDA")
    save_dtype = FLOAT_DTYPES[config.output.save_precision]
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    manifests = {
        name: load_minimax_h3_manifest(Path(data.cache_directory)) for name, data in config.dataset.groups.items()
    }
    formats = {
        name: MiniMaxH3CacheFormat.model_validate(manifest.model_dump(exclude={"entries"}))
        for name, manifest in manifests.items()
    }
    for name, data in config.dataset.groups.items():
        expected_layout = "single-frame" if data.kind == "image" else "video"
        if manifests[name].task != config.model.variant or manifests[name].frame_layout != expected_layout:
            raise ValueError("MiniMaxH3 cache task/frame layout differs from model variant and dataset kind")
    plan = DatasetBatches(
        {
            name: build_minimax_h3_batches(
                manifests[name], batch_size=data.batch_size, repeats=data.repeats, seed=config.train.seed
            )
            for name, data in config.dataset.groups.items()
        },
        seed=config.train.seed,
    )
    previews: AudioVideoPreviews | None = None
    if config.preview is not None:
        previews = import_module("unituner.training.integrations.minimax_h3.previews").AudioVideoPreviews(config)
    representation_targets = None if config.method.repa is None else RepaTargets(config.method.repa)
    if representation_targets is not None:
        for manifest in manifests.values():
            representation_targets.validate_sources(manifest.entries)
    identity = build_resume_identity(
        config,
        dataset_manifest_sha256=representation_dataset_sha256(
            canonical_sha256(
                [
                    {"group": name, "manifest": file_sha256(Path(data.cache_directory) / "manifest.json")}
                    for name, data in config.dataset.groups.items()
                ]
            ),
            representation_targets,
        ),
        model_artifact_sha256=weight_artifact_sha256(Path(config.model.weights)),
    )
    torch.manual_seed(config.train.seed)
    load_device = (
        torch.device("cpu") if execution.quantization != "none" or execution.offloads_frozen_weights else device
    )
    model = load_minimax_h3(
        Path(config.model.weights),
        device=load_device,
        dtype=training_base_dtype(execution, adapter=adapter is not None),
    )
    if model.config.text_dim != 5120:
        raise ValueError("MiniMaxH3 training requires native5120-wide cached student text")
    prepare_base_weights(model, execution, family=config.model.family.value, device=device)
    if execution.residency == "resident" and load_device != device:
        model.to(device=device)
    flow_config = config.method.minimax_h3 or MiniMaxH3FlowConfig()
    if config.train.timestep_sampling == "uniform":
        flow_config = flow_config.model_copy(update={"video_shift": 1.0, "audio_shift": 1.0})
    binding = MiniMaxH3Binding(model, flow_config)
    with TrainingSession() as session:
        alignment = install_objectives(
            model, binding.regions, config.method, session=session, targets=representation_targets, device=device
        )
        if previews is not None:
            session.own(previews)
        installation = session.own(
            ParameterInstallation(
                model,
                adapter=adapter,
                targets=binding.adapter_targets() if adapter is not None else None,
                adapter_device=device,
                adapter_dtype=torch.float32,
                include_layers=config.train.include_layers,
                exclude_layers=config.train.exclude_layers,
            )
        )
        if config.execution.offloads_frozen_weights:
            session.own(FrozenModelPlacement(model, binding.regions, device=device))
        optimizer = create_adamw(
            installation.plan.groups
            + (
                (ParameterGroup("repa", tuple(alignment.projection.parameters())),)
                if alignment is not None and alignment.projection is not None
                else ()
            ),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            config=config.train.optimizer,
        )
        record_context(config, installation.plan.target_names, identity)
        modules = {"model": model, "adapter": installation.branches}
        if alignment is not None and alignment.projection is not None:
            modules["repa"] = alignment.projection
        state = RunState()
        if resume is not None:
            cursor = restore_checkpoint(
                resume, modules, optimizer, identity, cuda_devices=(device.index or 0,) if device.type == "cuda" else ()
            )
            state = RunState(**cursor.model_dump())
            if (
                state.microbatches != state.updates * config.train.accumulation_steps
                or state.samples != plan.samples_before(state.microbatches)
                or state.updates > config.train.max_updates
            ):
                raise ValueError("MiniMaxH3 checkpoint cursor differs from the deterministic batch plan")
        session.own(ExecutionInstallation(binding.regions, execution, device=device))
        consumed = state.microbatches

        def steps() -> Iterator[StepResult]:
            """Read one complete sample and draw its native independent training noise."""
            nonlocal consumed
            for _ in range(config.train.accumulation_steps):
                name, entries = plan.batch(consumed)
                cache = Path(config.dataset.groups[name].cache_directory)
                format = formats[name]
                record = read_minimax_h3_record(cache, entries[0], format=format, device=device)
                if alignment is not None:
                    alignment.set_targets(entries, record.video.unsqueeze(0))
                request = sample_minimax_h3_request(record, format=format, config=binding.config)
                consumed += 1
                yield binding.step(request)

        def write_preview(update: int) -> None:
            """Borrow current weights after the shared update/checkpoint boundary."""
            if previews is not None:
                previews(update, model, installation.branches)

        def write_weights(update: int) -> None:
            """Export portable weights without optimizer or RNG state."""
            if adapter is None:
                save_minimax_h3(
                    model,
                    weights_path(config, update=update),
                    dtype=save_dtype,
                    metadata=publication_metadata(config, identity, update),
                )
            else:
                publish_adapter(
                    weights_path(config, update=update),
                    branches=installation.branches,
                    target_names=installation.plan.target_names,
                    adapter=adapter,
                    base_model="minimax_h3",
                    module_identifier="networks.lora_minimax_h3",
                    dtype=save_dtype,
                    namespace="diffusion_model",
                    adapter_format=config.output.adapter_format,
                    family="minimax-h3",
                    metadata={
                        **publication_metadata(config, identity, update),
                        "model_variant": config.model.variant,
                        "unituner.adapter.layout": "minimax-h3-comfyui",
                        "unituner.minimax_h3.flow": binding.config.model_dump_json(),
                    },
                )

            publish_latest_weights(config, update)

        run_updates(
            config,
            lambda: align_steps(steps, alignment),
            optimizer=optimizer,
            modules=modules,
            state=state,
            identity=identity,
            device=device,
            weights_writer=write_weights,
            preview_writer=None if previews is None else write_preview,
        )
        if not weights_path(config).is_file():
            write_weights(state.updates)
    return state
