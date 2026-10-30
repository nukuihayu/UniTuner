"""Assemble native image flow/restoration training with explicit storage and compute precision from explicit native cached data."""

import json
from collections.abc import Callable, Generator, Iterator
from contextlib import closing
from pathlib import Path
from time import monotonic

import torch
from pydantic import JsonValue
from torch import nn

from unituner.analysis.context import publication_metadata, record_context
from unituner.artifacts.hashing import canonical_sha256
from unituner.artifacts.tensors import save_tensors, weight_artifact_sha256
from unituner.checkpoints.restore import restore_checkpoint
from unituner.checkpoints.schema import build_resume_identity
from unituner.config.training import TrainingConfig
from unituner.data.cache.representation import RepaTargets, representation_dataset_sha256
from unituner.data.cache.storage import read_latent_record
from unituner.data.datasets.batching import DatasetBatches, EpochBatches
from unituner.data.datasets.cached_images import index_cached_images
from unituner.data.datasets.restoration import RestorationDataset
from unituner.execution.frozen_blocks import FrozenModelPlacement
from unituner.execution.installation import (
    ExecutionInstallation,
    prepare_base_weights,
    training_base_dtype,
    validate_training_execution,
)
from unituner.execution.precision import resolve_training_precision
from unituner.execution.runtime import runtime_scope
from unituner.logging import set_context
from unituner.models.catalog import ModelFamily
from unituner.models.components.conditioning import batch_captions
from unituner.models.flux.transformer import FluxTransformer
from unituner.models.image import ImageModel, load_image_model
from unituner.models.qwen_image_2_1.data.cache.storage import read_qwen_image_21_record
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unituner.precision import FLOAT_DTYPES
from unituner.progress import Progress, status
from unituner.training.contracts import ImageFlowRequest, RunState, StepResult
from unituner.training.engine import run_updates
from unituner.training.integrations.flux.binding import FluxBinding
from unituner.training.integrations.flux2.binding import Flux2Binding, Flux2FlowRequest
from unituner.training.integrations.flux3.binding import Flux3Binding, Flux3FlowRequest
from unituner.training.integrations.image_models import bind_image_model
from unituner.training.integrations.image_publication import publish_image_model
from unituner.training.integrations.krea2.binding import Krea2Binding, Krea2FlowRequest
from unituner.training.integrations.qwen_image.binding import QwenImageBinding
from unituner.training.integrations.qwen_image_2_1.binding import QwenImage21Binding
from unituner.training.losses.wavelet import WaveletLoss
from unituner.training.methods.restoration import RestorationMethod
from unituner.training.objectives import align_steps, install_objectives
from unituner.training.optimizers.adamw import create_adamw
from unituner.training.output import publish_latest_weights, weights_path
from unituner.training.parameters import ParameterGroup, ParameterInstallation
from unituner.training.session import TrainingSession
from unituner.training.timesteps import sample_sigmas, training_sigma_grid

type PreviewWriter = Callable[
    [
        int,
        ImageModel,
        nn.Module,
        RestorationMethod | None,
    ],
    None,
]


@runtime_scope
def train(
    config: TrainingConfig,
    *,
    device: torch.device,
    resume: Path | None = None,
    restoration_data: RestorationDataset | None = None,
    preview_writer: PreviewWriter | None = None,
) -> RunState:
    """Run one native cached-image training session to the configured update limit.

    Support native image cached flow and Z-Image/Qwen/Krea2 restoration,
    FP32 full masters or BF16/FP32 frozen bases with FP32 adapters,
    optional block recompute and Inductor. Reject all other explicit choices
    before artifact IO. Shape buckets shuffle within each epoch and retain short
    tails; the microbatch cursor reconstructs the exact data order on resume. Save native
    optimizer/RNG checkpoints and full or published-key adapter weights.
    """
    if (config.preview is not None) != (preview_writer is not None):
        raise ValueError("Configured previews require an explicit image preview provider")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Training entry currently supports CPU or CUDA devices")
    config = resolve_training_precision(config)
    execution = config.execution
    restoration = config.method.kind == "restoration"
    if config.model.family not in {
        ModelFamily.Z_IMAGE,
        ModelFamily.MOSS_IMAGE,
        ModelFamily.FLUX,
        ModelFamily.FLUX3,
        ModelFamily.FLUX2,
        ModelFamily.QWEN_IMAGE,
        ModelFamily.QWEN_IMAGE_2_1,
        ModelFamily.KREA2,
    } or any(data.kind != ("restoration" if restoration else "image") for data in config.dataset.groups.values()):
        raise ValueError("Training entry supports native cached image flow and Z-Image/Qwen/Krea2 restoration")
    if restoration != (restoration_data is not None):
        raise ValueError("Restoration requires an explicitly indexed image restoration dataset")
    if restoration and any(
        min(data.resolution) < 16 or any(value % 16 for value in data.resolution)
        for data in config.dataset.groups.values()
    ):
        raise ValueError("Restoration resolution must be at least 16 and divisible by 16")
    validate_training_execution(execution, device=device, adapter=config.train.adapter is not None)
    precision = execution.precision
    if precision == "auto":
        raise ValueError("Training precision must be resolved before running updates")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Training execution requires CPU/CUDA")
    save_dtype = FLOAT_DTYPES[config.output.save_precision]
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(config.train.seed)
    sigma_grid = (
        None if restoration else training_sigma_grid(config.model.family, config.train.timestep_sampling, device=device)
    )
    if not restoration and config.train.skip_dataset_validation:
        status("[train] Indexing trusted image caches (full dataset validation skipped)")
    cached_groups, batch_sizes, dataset_digest = (
        ({}, {}, "")
        if restoration
        else index_cached_images(
            config.dataset,
            family=config.model.family,
            variant=config.model.variant,
            version=config.model.version,
            skip_validation=config.train.skip_dataset_validation,
            validation_workers=config.train.dataset_validation_workers,
        )
    )
    cached_plan = (
        None
        if restoration
        else EpochBatches(
            cached_groups,
            batch_size=batch_sizes,
            seed=config.train.seed,
        )
    )
    restoration_plan = (
        None
        if restoration_data is None
        else DatasetBatches(
            {
                name: EpochBatches(
                    {
                        (0,): tuple(
                            pair
                            for pair in restoration_data.pairs
                            if pair.cache_directory == Path(data.cache_directory)
                        )
                    },
                    batch_size=data.batch_size,
                    seed=config.train.seed,
                )
                for name, data in config.dataset.groups.items()
            },
            seed=config.train.seed,
        )
    )
    status("[train] Hashing model artifacts for checkpoint identity (large files may take a while)")
    artifact_digest = weight_artifact_sha256(Path(config.model.weights), release_cache=restoration)
    if restoration:
        paths: dict[str, JsonValue] = {"transformer": artifact_digest}
        if config.model.vae is None or config.method.restoration is None:
            raise ValueError("Restoration requires a vae and method settings")
        for name, path in (
            ("vae", config.model.vae),
            ("dino", config.method.restoration.dino_weights),
            ("vgg", config.method.restoration.vgg_weights),
        ):
            if path is not None:
                paths[name] = weight_artifact_sha256(Path(path), release_cache=True)
        artifact_digest = canonical_sha256(paths)
    representation_targets = None if config.method.repa is None else RepaTargets(config.method.repa)
    if representation_targets is not None:
        for group in cached_groups.values():
            representation_targets.validate_sources(tuple(item.entry for item in group))
    identity = build_resume_identity(
        config,
        dataset_manifest_sha256=representation_dataset_sha256(
            restoration_data.digest if restoration_data is not None else dataset_digest,
            representation_targets,
        ),
        model_artifact_sha256=artifact_digest,
    )
    load_device = (
        torch.device("cpu") if execution.quantization != "none" or execution.offloads_frozen_weights else device
    )
    base_dtype = training_base_dtype(execution, adapter=config.train.adapter is not None)
    storage = {torch.bfloat16: "BF16", torch.float16: "FP16", torch.float32: "FP32"}[base_dtype]
    status(f"[train] Loading denoiser weights: {config.model.weights} -> {load_device} ({storage} base storage)")
    model = load_image_model(config.model, device=load_device, dtype=base_dtype)
    binding = bind_image_model(model)
    status(
        f"[train] Denoiser loaded; preparing base weights (quantization={execution.quantization}, residency={execution.residency})"
    )
    prepare_base_weights(model, execution, family=config.model.family.value, device=device)
    if execution.residency == "resident" and load_device != device:
        model.to(device=device)
    if isinstance(model, FluxTransformer) and config.method.guidance is not None and not model.config.guidance_embed:
        raise ValueError("Explicit method.guidance requires Flux guidance embedding")
    if not isinstance(model, QwenImage21Transformer) and model.config.variant != config.model.variant:
        raise ValueError("Training model variant differs from native artifact topology")
    output = Path(config.output.directory)
    output.mkdir(parents=True, exist_ok=True)
    if config.method.wavelet is not None and not restoration:
        binding.wavelet = WaveletLoss(config.method.wavelet, device=device)
    with TrainingSession() as session:
        alignment = install_objectives(
            model, binding.regions, config.method, session=session, targets=representation_targets, device=device
        )
        status("[train] Installing trainable parameters/adapters")
        installation = session.own(
            ParameterInstallation(
                model,
                adapter=config.train.adapter,
                adapter_device=device,
                adapter_dtype=torch.float32,
                include_layers=config.train.include_layers,
                exclude_layers=config.train.exclude_layers,
                targets=binding.adapter_targets() if config.train.adapter is not None else None,
            )
        )
        if config.execution.offloads_frozen_weights:
            session.own(FrozenModelPlacement(model, binding.regions, device=device))
        if restoration:
            method = RestorationMethod(config, binding, device, session)
        else:
            method = None
        generator_groups = installation.plan.groups
        if method is not None and method.encoder is not None:
            generator_groups += (ParameterGroup("encoder", tuple(method.encoder.parameters())),)
        if alignment is not None and alignment.projection is not None:
            generator_groups += (ParameterGroup("repa", tuple(alignment.projection.parameters())),)
        status(
            f"[train] Creating optimizer; generator trainable parameters={sum(p.numel() for group in generator_groups for p in group.parameters):,}"
        )
        optimizer_started = monotonic()
        set_context(stage="creating generator optimizer")
        optimizer = create_adamw(
            generator_groups,
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            config=config.train.optimizer,
        )
        status(
            f"[train] Generator optimizer ready in {monotonic() - optimizer_started:.2f}s; Adam states allocate on first update"
        )
        optimizers = {"generator" if restoration else "main": optimizer}
        modules: dict[str, nn.Module] = {"model": model, "adapter": installation.branches}
        if alignment is not None and alignment.projection is not None:
            modules["repa"] = alignment.projection
        if method is not None:
            modules.update(method.modules)
            if method.discriminator is not None:
                status("[train] Creating discriminator optimizer")
                optimizers["discriminator"] = create_adamw(
                    (ParameterGroup("heads", tuple(method.discriminator.heads.parameters())),),
                    lr=config.train.learning_rate,
                    weight_decay=config.train.weight_decay,
                    config=config.train.optimizer,
                )
        status("[train] Recording training context")
        set_context(stage="recording training context")
        record_context(
            config,
            installation.plan.target_names,
            identity,
            encoder_targets=method.encoder.names if method is not None and method.encoder is not None else (),
        )
        state = RunState()
        if resume is not None:
            status(f"[train] Restoring checkpoint: {resume}")
            cursor = restore_checkpoint(
                resume,
                modules,
                optimizers if restoration else optimizer,
                identity,
                cuda_devices=(device.index or 0,) if device.type == "cuda" else (),
            )
            state = RunState(**cursor.model_dump())
        elif method is not None and restoration_data is not None and config.method.restoration is not None:
            if config.method.restoration.calibrate:
                method.calibrate(config, restoration_data)
        if method is not None:
            if not 0 < method.sigma < 1:
                raise ValueError("Restored method sigma must be finite and strictly inside (0,1)")
            print(json.dumps({"restoration_sigma": method.sigma}), flush=True)
        status(f"[train] Installing execution policy: recompute={execution.recompute}, compile={execution.compile}")
        session.own(ExecutionInstallation(binding.regions, execution, device=device))
        consumed = state.samples
        microbatch = state.microbatches
        plan = restoration_plan if restoration_plan is not None else cached_plan
        if plan is None or consumed != plan.samples_before(microbatch):
            raise ValueError("Checkpoint sample cursor differs from the dataset batch schedule")

        def steps() -> Iterator[StepResult]:
            """Stream graphs one microbatch at a time from the exact consumed-sample cursor."""
            nonlocal consumed, microbatch
            for microbatch_index in range(config.train.accumulation_steps):
                set_context(
                    update=state.updates + 1, microbatch=microbatch_index + 1, phase="forward", stage="training"
                )
                if restoration_data is not None and method is not None:
                    pairs = []
                    selected_pairs = []
                    if restoration_plan is None:
                        raise RuntimeError("Restoration requires its batch schedule")
                    group_name, batch_pairs = restoration_plan.batch(microbatch)
                    set_context(group=group_name, source=", ".join(str(pair.high) for pair in batch_pairs))
                    data = config.dataset.groups[group_name]
                    for offset, pair in enumerate(batch_pairs):
                        pairs.append(
                            restoration_data.read(
                                pair,
                                data.resolution,
                                config.train.seed + consumed + offset,
                                data.horizontal_flip_probability,
                            )
                        )
                        selected_pairs.append(pair)
                    microbatch += 1
                    consumed += len(pairs)
                    low_batch = (
                        pairs[0][0].unsqueeze(0) if len(pairs) == 1 else torch.stack([p[0] for p in pairs])
                    ).to(device)
                    high_batch = (
                        pairs[0][1].unsqueeze(0) if len(pairs) == 1 else torch.stack([p[1] for p in pairs])
                    ).to(device)
                    del pairs
                    yield from method.steps(
                        low_batch,
                        high_batch,
                        method.captions(selected_pairs),
                    )
                    del low_batch, high_batch
                    continue
                if cached_plan is None:
                    raise RuntimeError("Image flow requires its batch schedule")
                selected = cached_plan.batch(microbatch)
                microbatch += 1
                if isinstance(binding, QwenImage21Binding):
                    records21 = [
                        read_qwen_image_21_record(item.directory, item.entry, device=device) for item in selected
                    ]
                    latents21 = torch.stack([record.latents for record in records21])
                    if alignment is not None:
                        alignment.set_targets(
                            tuple(item.entry for item in selected),
                            latents21,
                        )
                    consumed += latents21.shape[0]
                    yield binding.cached_step(
                        records21,
                        torch.randn_like(latents21),
                        sample_sigmas(latents21.shape[0], device=device, grid=sigma_grid),
                    )
                    continue
                records = [read_latent_record(item.directory, item.entry, device=device) for item in selected]
                latents = torch.stack([record.latents for record in records])
                if alignment is not None:
                    alignment.set_targets(
                        tuple(item.entry for item in selected),
                        latents,
                        skip_first_frame=isinstance(binding, QwenImageBinding)
                        and config.method.qwen_image is not None
                        and config.method.qwen_image.remove_base,
                    )
                consumed += latents.shape[0]
                noise = torch.randn_like(latents)
                sigma = sample_sigmas(latents.shape[0], device=device, grid=sigma_grid)
                if isinstance(binding, QwenImageBinding):
                    yield binding.cached_step(
                        records,
                        noise,
                        sigma,
                        remove_base=config.method.qwen_image is not None and config.method.qwen_image.remove_base,
                    )
                    continue
                if isinstance(binding, FluxBinding):
                    yield binding.cached_step(
                        records,
                        noise,
                        sigma,
                        guidance=3.5 if config.method.guidance is None else config.method.guidance,
                    )
                    continue
                if isinstance(binding, Flux2Binding):
                    yield binding.step(
                        Flux2FlowRequest(
                            latents,
                            noise,
                            sigma,
                            tuple(record.caption for record in records),
                            tuple(record.references for record in records),
                            guidance=1.0 if config.method.guidance is None else config.method.guidance,
                        )
                    )
                    continue
                if isinstance(binding, Flux3Binding):
                    yield binding.step(
                        Flux3FlowRequest(
                            latents,
                            noise,
                            sigma,
                            tuple(record.caption for record in records),
                            tuple(record.references for record in records),
                        )
                    )
                    continue
                if isinstance(binding, Krea2Binding):
                    yield binding.step(
                        Krea2FlowRequest(latents, noise, sigma, tuple(record.caption for record in records))
                    )
                    continue
                caption = batch_captions([record.caption for record in records])
                yield binding.step(
                    ImageFlowRequest(
                        latents,
                        noise,
                        sigma,
                        caption.features,
                        caption.mask,
                        tuple(
                            torch.stack([record.references[index] for record in records])
                            for index in range(len(records[0].references))
                        ),
                    )
                )

        def monitored_steps() -> Generator[StepResult]:
            """Advance accumulation progress only after the microbatch's final backward."""
            last_phase = (
                "discriminator"
                if method is not None and method.discriminator is not None
                else "generator"
                if restoration
                else "main"
            )
            with closing(
                Progress(
                    "[train] accumulation",
                    config.train.accumulation_steps,
                    unit="microbatch",
                    position=1,
                    leave=False,
                )
            ) as progress:
                completed = 0
                for result in align_steps(steps, alignment):
                    yield result
                    if result.phase == last_phase:
                        completed += 1
                        progress.report(completed)

        def write_weights(update: int) -> Path:
            """Export inference weights and the separate restoration encoder."""
            artifact = weights_path(config, update=update)
            method_metadata = (
                {}
                if method is None
                else {
                    "unituner.restoration.sigma": repr(method.sigma),
                    "unituner.restoration.posterior": method.posterior,
                }
            )
            method_metadata.update(publication_metadata(config, identity, update))
            if method is not None and method.encoder is not None:
                save_tensors(
                    method.encoder.state_dict(),
                    weights_path(config, update=update, encoder=True),
                    dtype=save_dtype,
                    metadata={
                        **method_metadata,
                        "unituner.encoder.targets": json.dumps(method.encoder.names),
                        "unituner.encoder.rank": str(config.method.restoration.encoder_rank)
                        if config.method.restoration is not None
                        else "0",
                    },
                )
            publish_image_model(
                config,
                model,
                binding,
                installation,
                artifact,
                dtype=save_dtype,
                metadata=method_metadata,
            )
            publish_latest_weights(config, update)
            return artifact

        def preview(update: int) -> None:
            if preview_writer is not None:
                preview_writer(update, model, installation.branches, method)

        run_updates(
            config,
            monitored_steps,
            optimizer=optimizers if restoration else optimizer,
            modules=modules,
            state=state,
            identity=identity,
            device=device,
            weights_writer=write_weights,
            preview_writer=preview if preview_writer is not None else None,
        )
        artifact = weights_path(config)
        if not artifact.is_file():
            write_weights(state.updates)
    status(f"[train] Complete; saved {artifact}")
    return state
