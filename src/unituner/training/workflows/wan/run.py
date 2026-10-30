"""Run native cached Wan full/adapter training using shared engine, resource and checkpoint owners."""

from collections.abc import Callable, Iterator
from pathlib import Path

import torch
from torch import nn

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
from unituner.models.wan.data.batching import build_wan_batches
from unituner.models.wan.data.cache.storage import load_wan_manifest, read_wan_record
from unituner.models.wan.loading import load_wan, save_wan
from unituner.models.wan.transformer import WanTransformer
from unituner.precision import FLOAT_DTYPES
from unituner.training.contracts import RunState, StepResult
from unituner.training.engine import run_updates
from unituner.training.integrations.wan.binding import WanBinding
from unituner.training.objectives import align_steps, install_objectives
from unituner.training.optimizers.adamw import create_adamw
from unituner.training.output import publish_latest_weights, weights_path
from unituner.training.parameters import ParameterGroup, ParameterInstallation
from unituner.training.publishing import publish_adapter
from unituner.training.session import TrainingSession
from unituner.training.timesteps import sample_sigmas, training_sigma_grid


@runtime_scope
def train_wan(
    config: TrainingConfig,
    *,
    device: torch.device,
    resume: Path | None = None,
    preview_writer: Callable[[int, WanTransformer, nn.Module], None] | None = None,
) -> RunState:
    """Train one resident FP32 Wan expert from immutable image/video mean caches.

    Use shared full/adapter AdamW, optional block recompute, sample-correct
    accumulation and deterministic stable-grid buckets including short tails.
    Model-specific FP32 sigma grids keep noise fractions and native times paired.
    Restore optimizer/RNG/data cursor before the next draw. Publish requested
    transformer.* LoRA or lycoris_* LoHa/LoKr names with alpha and unprefixed
    metadata. Raw names alone do not establish external ComfyUI compatibility.

    Reject unaccepted compile/precision/residency/expert configurations
    before weight IO. Frozen base parameters are included in resume snapshots;
    output artifacts contain the selected full weights or adapters. No RCM or posterior sampling occurs.
    """
    config = resolve_training_precision(config)
    execution = config.execution
    adapter = config.train.adapter
    if (
        config.model.family != ModelFamily.WAN
        or any(data.kind not in ("image", "video") for data in config.dataset.groups.values())
        or config.method.kind != "flow-matching"
    ):
        raise ValueError("Wan training requires cached image/video flow in full or adapter mode")
    if (config.preview is not None) != (preview_writer is not None):
        raise ValueError("Wan configured previews require an explicit video preview writer")
    if config.method.guidance is not None:
        raise ValueError("Wan training guidance embedding is not supported")
    validate_training_execution(execution, device=device, adapter=config.train.adapter is not None)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Training execution requires CPU/CUDA")
    save_dtype = FLOAT_DTYPES[config.output.save_precision]
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    manifests = {name: load_wan_manifest(Path(data.cache_directory)) for name, data in config.dataset.groups.items()}
    for manifest in manifests.values():
        if manifest.version != config.model.version or manifest.variant != config.model.variant:
            raise ValueError("Wan cache version/variant differs from the training model")
        if manifest.image_dim is not None and manifest.image_dim != 1280:
            raise ValueError("Wan transformer training requires1280-wide CLIP features")
        if manifest.frame_layout == "single-frame" and manifest.clean_count != 1 and manifest.version == "2.1":
            raise ValueError("Wan2.1 two-clean training requires explicit FLF2V application support")
    plan = DatasetBatches(
        {
            name: build_wan_batches(
                Path(data.cache_directory),
                manifests[name],
                batch_size=data.batch_size,
                repeats=data.repeats,
                seed=config.train.seed,
            )
            for name, data in config.dataset.groups.items()
        },
        seed=config.train.seed,
    )
    representation_targets = None if config.method.repa is None else RepaTargets(config.method.repa)
    if representation_targets is not None:
        for manifest in manifests.values():
            representation_targets.validate_sources(manifest.entries)
    dataset_digest = canonical_sha256(
        [
            {"group": name, "manifest": file_sha256(Path(data.cache_directory) / "manifest.json")}
            for name, data in config.dataset.groups.items()
        ]
    )
    identity = build_resume_identity(
        config,
        dataset_manifest_sha256=representation_dataset_sha256(dataset_digest, representation_targets),
        model_artifact_sha256=weight_artifact_sha256(Path(config.model.weights)),
    )
    torch.manual_seed(config.train.seed)
    sigma_grid = training_sigma_grid("wan", config.train.timestep_sampling, device=device)
    load_device = (
        torch.device("cpu") if execution.quantization != "none" or execution.offloads_frozen_weights else device
    )
    model = load_wan(
        Path(config.model.weights),
        variant=config.model.variant,
        version=config.model.version,
        device=load_device,
        dtype=training_base_dtype(execution, adapter=adapter is not None),
    )
    for manifest in manifests.values():
        if (
            model.config.version != manifest.version
            or model.config.variant != manifest.variant
            or model.config.text_dim != manifest.text_dim
        ):
            raise ValueError("Wan model topology differs from cache version/variant/text width")
        expected_channels = 48 if manifest.control else 36 if manifest.variant == "i2v" else 16
        if model.config.in_channels != expected_channels or model.config.out_channels != 16:
            raise ValueError("Wan model latent/control input channels differ from cache geometry")
    prepare_base_weights(model, execution, family=config.model.family.value, device=device)
    if execution.residency == "resident" and load_device != device:
        model.to(device=device)
    binding = WanBinding(model)
    output = Path(config.output.directory)
    output.mkdir(parents=True, exist_ok=True)
    with TrainingSession() as session:
        alignment = install_objectives(
            model, binding.regions, config.method, session=session, targets=representation_targets, device=device
        )
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
            ):
                raise ValueError("Wan checkpoint cursor differs from the deterministic batch plan")
        session.own(ExecutionInstallation(binding.regions, execution, device=device))
        consumed = state.microbatches

        def steps() -> Iterator[StepResult]:
            """Read each equal-grid batch and draw noise followed by its training sigma."""
            nonlocal consumed
            for _ in range(config.train.accumulation_steps):
                name, entries = plan.batch(consumed)
                cache = Path(config.dataset.groups[name].cache_directory)
                records = [read_wan_record(cache, entry, manifest=manifests[name], device=device) for entry in entries]
                latents = torch.stack([record.latents for record in records])
                if alignment is not None:
                    alignment.set_targets(tuple(entries), latents)
                noise = torch.randn_like(latents)
                sigma = sample_sigmas(latents.shape[0], device=device, grid=sigma_grid)
                timesteps = sigma * 1000
                if sigma_grid is None:
                    timesteps = timesteps + 1
                consumed += 1
                yield binding.cached_step(records, noise, sigma, timesteps)

        def preview(update: int) -> None:
            """Bind Wan-specific resources to the shared update loop's preview boundary."""
            if preview_writer is not None:
                preview_writer(update, model, installation.branches)

        def write_weights(update: int) -> None:
            """Export portable weights without optimizer or RNG state."""
            if adapter is None:
                save_wan(
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
                    base_model="wan" + str(config.model.version),
                    module_identifier="networks.lora_wan",
                    dtype=save_dtype,
                    namespace="diffusion_model",
                    adapter_format=config.output.adapter_format,
                    family="wan",
                    metadata={
                        **publication_metadata(config, identity, update),
                        "model_version": str(config.model.version),
                        "model_variant": config.model.variant,
                        "unituner.adapter.layout": "wan-comfyui",
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
            preview_writer=preview if preview_writer is not None else None,
        )
        if not weights_path(config).is_file():
            write_weights(state.updates)
    return state
