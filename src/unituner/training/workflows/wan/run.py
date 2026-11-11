"""Run native cached Wan full/adapter training using shared engine, resource and checkpoint owners."""

from collections.abc import Callable, Iterator
from pathlib import Path

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
from unituner.models.wan.data.batching import build_wan_batches
from unituner.models.wan.data.cache.storage import load_wan_manifest, read_wan_record
from unituner.models.wan.loading import load_wan, save_wan
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


def _choose_high_noise(*, grid_size: int | None, high_size: int | None, boundary: float) -> bool:
    """Choose on the host without synchronizing the accelerator's sigma stream."""
    if grid_size is None:
        return bool(torch.rand(()).item() >= boundary - 0.001)
    if high_size is None or high_size < 1 or high_size >= grid_size:
        raise ValueError("Wan sigma grid must include both expert domains")
    return bool(torch.randint(grid_size, ()).item() < high_size)


def _sample_expert_batch(
    count: int,
    *,
    device: torch.device,
    grid: torch.Tensor | None,
    boundary: float,
    low_grid: torch.Tensor | None,
    high_grid: torch.Tensor | None,
) -> tuple[torch.Tensor, bool]:
    """Select one expert with its original probability and sample the rest conditionally."""
    high = _choose_high_noise(
        grid_size=None if grid is None else grid.numel(),
        high_size=None if high_grid is None else high_grid.numel(),
        boundary=boundary,
    )
    if grid is not None:
        domain = high_grid if high else low_grid
        if domain is None or domain.numel() == 0:
            raise ValueError("Wan expert sigma grid has no values in the selected domain")
        sigmas = sample_sigmas(count, device=device, grid=domain)
    else:
        threshold = boundary - 0.001  # Native uniform training times are sigma*1000 + 1.
        draws = torch.rand(count, device=device)
        sigmas = threshold + (1 - threshold) * draws if high else threshold * draws
    return sigmas, high


@runtime_scope
def train_wan(
    config: TrainingConfig,
    *,
    device: torch.device,
    resume: Path | None = None,
    preview_writer: Callable[..., None] | None = None,
) -> RunState:
    """Train one Wan expert or a paired Wan2.2 high/low-noise set from immutable caches.

    Use shared full/adapter AdamW, optional block recompute, sample-correct
    accumulation and deterministic stable-grid buckets including short tails.
    Model-specific FP32 sigma grids keep noise fractions and native times paired.
    Restore optimizer/RNG/data cursor before the next draw. Publish requested
    transformer.* LoRA or lycoris_* LoHa/LoKr names with alpha and unprefixed
    metadata. Raw names alone do not establish external ComfyUI compatibility.

    Paired experts use one expert per microbatch, preserving each sample's
    marginal noise distribution. Both models, adapters and optimizer states
    resume together.
    Frozen base parameters are included in resume snapshots; exports contain
    separate full weights or adapters. No RCM or posterior sampling occurs.
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
    high_settings = config.wan_experts
    artifact_digest = weight_artifact_sha256(Path(config.model.weights))
    if high_settings is not None:
        artifact_digest = canonical_sha256(
            {"low": artifact_digest, "high": weight_artifact_sha256(Path(high_settings.high_noise_weights))}
        )
    identity = build_resume_identity(
        config,
        dataset_manifest_sha256=representation_dataset_sha256(dataset_digest, representation_targets),
        model_artifact_sha256=artifact_digest,
    )
    torch.manual_seed(config.train.seed)
    sigma_grid = training_sigma_grid("wan", config.train.timestep_sampling, device=device)
    low_grid = high_grid = None
    if high_settings is not None:
        if sigma_grid is None:
            if high_settings.boundary <= 0.001:
                raise ValueError("Wan uniform sigma sampling cannot reach the low-noise expert at this boundary")
        else:
            high_mask = sigma_grid * 1000 >= high_settings.boundary * 1000
            high_grid, low_grid = sigma_grid[high_mask], sigma_grid[~high_mask]
            if not high_grid.numel() or not low_grid.numel():
                raise ValueError("Wan sigma grid must include both expert domains at the configured boundary")
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
    high_model = (
        None
        if high_settings is None
        else load_wan(
            Path(high_settings.high_noise_weights),
            variant=config.model.variant,
            version=config.model.version,
            device=load_device,
            dtype=training_base_dtype(execution, adapter=adapter is not None),
        )
    )
    if high_model is not None and high_model.config != model.config:
        raise ValueError("Wan2.2 experts require identical native topology")
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
    if high_model is not None:
        prepare_base_weights(high_model, execution, family=config.model.family.value, device=device)
    if execution.residency == "resident" and load_device != device:
        model.to(device=device)
        if high_model is not None:
            high_model.to(device=device)
    binding = WanBinding(model)
    high_binding = None if high_model is None else WanBinding(high_model)
    output = Path(config.output.directory)
    output.mkdir(parents=True, exist_ok=True)
    with TrainingSession() as session:
        alignment = install_objectives(
            model, binding.regions, config.method, session=session, targets=representation_targets, device=device
        )
        high_alignment = (
            None
            if high_binding is None
            else install_objectives(
                high_model,
                high_binding.regions,
                config.method,
                session=session,
                targets=representation_targets,
                device=device,
            )
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
        high_installation = (
            None
            if high_binding is None
            else session.own(
                ParameterInstallation(
                    high_model,
                    adapter=adapter,
                    targets=high_binding.adapter_targets() if adapter is not None else None,
                    adapter_device=device,
                    adapter_dtype=torch.float32,
                    include_layers=config.train.include_layers,
                    exclude_layers=config.train.exclude_layers,
                )
            )
        )
        if config.execution.offloads_frozen_weights:
            session.own(FrozenModelPlacement(model, binding.regions, device=device))
            if high_binding is not None:
                session.own(FrozenModelPlacement(high_model, high_binding.regions, device=device))
        groups = (*installation.plan.groups,)
        if high_installation is not None:
            groups += tuple(
                ParameterGroup(f"high_noise.{group.name}", group.parameters, group.learning_rate_scale)
                for group in high_installation.plan.groups
            )
        for name, owner in (("repa", alignment), ("high_noise.repa", high_alignment)):
            if owner is not None and owner.projection is not None:
                groups += (ParameterGroup(name, tuple(owner.projection.parameters())),)
        optimizer = create_adamw(
            groups,
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            config=config.train.optimizer,
        )
        record_context(
            config,
            installation.plan.target_names
            + (
                ()
                if high_installation is None
                else tuple(f"high_noise.{name}" for name in high_installation.plan.target_names)
            ),
            identity,
        )
        modules = {"model": model, "adapter": installation.branches}
        if alignment is not None and alignment.projection is not None:
            modules["repa"] = alignment.projection
        if high_model is not None and high_installation is not None:
            modules.update(high_noise_model=high_model, high_noise_adapter=high_installation.branches)
        if high_alignment is not None and high_alignment.projection is not None:
            modules["high_noise_repa"] = high_alignment.projection
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
        if high_binding is not None:
            session.own(ExecutionInstallation(high_binding.regions, execution, device=device))
        consumed = state.microbatches

        def steps() -> Iterator[StepResult]:
            """Read each equal-grid batch and draw noise followed by its training sigma."""
            nonlocal consumed
            for _ in range(config.train.accumulation_steps):
                name, entries = plan.batch(consumed)
                cache = Path(config.dataset.groups[name].cache_directory)
                records = [read_wan_record(cache, entry, manifest=manifests[name], device=device) for entry in entries]
                latents = torch.stack([record.latents for record in records])
                if alignment is not None and high_settings is None:
                    alignment.set_targets(tuple(entries), latents)
                noise = torch.randn_like(latents)
                if high_settings is not None:
                    sigma, high = _sample_expert_batch(
                        len(records),
                        device=device,
                        grid=sigma_grid,
                        boundary=high_settings.boundary,
                        low_grid=low_grid,
                        high_grid=high_grid,
                    )
                else:
                    sigma = sample_sigmas(latents.shape[0], device=device, grid=sigma_grid)
                timesteps = sigma * 1000
                if sigma_grid is None:
                    timesteps = timesteps + 1
                consumed += 1
                if high_binding is None:
                    yield binding.cached_step(records, noise, sigma, timesteps, latents=latents)
                    continue
                expert = high_binding if high else binding
                objective = high_alignment if high else alignment
                if objective is None:
                    result = expert.cached_step(records, noise, sigma, timesteps, latents=latents)
                else:
                    with objective.capture():
                        objective.set_targets(tuple(entries), latents)
                        result = objective.step(expert.cached_step(records, noise, sigma, timesteps, latents=latents))
                label = "high_noise" if high else "low_noise"
                yield StepResult(
                    result.loss,
                    result.sample_count,
                    metrics={
                        f"expert/{label}_loss": result.loss.detach(),
                        f"expert/{label}_samples": result.loss.new_tensor(result.sample_count),
                        **{f"expert/{label}/{key}": value for key, value in result.metrics.items()},
                    },
                )

        def preview(update: int) -> None:
            """Bind Wan-specific resources to the shared update loop's preview boundary."""
            if preview_writer is not None:
                if high_model is None or high_installation is None:
                    preview_writer(update, model, installation.branches)
                else:
                    preview_writer(
                        update,
                        model,
                        installation.branches,
                        high_noise_model=high_model,
                        high_noise_branches=high_installation.branches,
                    )

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

            if high_model is not None and high_installation is not None:
                high_path = weights_path(config, update=update, expert="high")
                high_metadata = {**publication_metadata(config, identity, update), "unituner.wan.expert": "high_noise"}
                if adapter is None:
                    save_wan(high_model, high_path, dtype=save_dtype, metadata=high_metadata)
                else:
                    publish_adapter(
                        high_path,
                        branches=high_installation.branches,
                        target_names=high_installation.plan.target_names,
                        adapter=adapter,
                        base_model="wan" + str(config.model.version),
                        module_identifier="networks.lora_wan",
                        dtype=save_dtype,
                        namespace="diffusion_model",
                        adapter_format=config.output.adapter_format,
                        family="wan",
                        metadata={
                            **high_metadata,
                            "model_version": str(config.model.version),
                            "model_variant": config.model.variant,
                            "unituner.adapter.layout": "wan-comfyui",
                        },
                    )

            publish_latest_weights(config, update)

        run_updates(
            config,
            lambda: align_steps(steps, alignment) if high_binding is None else steps(),
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
