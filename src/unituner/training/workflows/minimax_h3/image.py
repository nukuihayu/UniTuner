"""Train H3-Image flow and RGB restoration from verified independent caches."""

import json
from collections.abc import Iterator
from pathlib import Path

import torch

from unituner.analysis.context import publication_metadata, record_context
from unituner.artifacts.hashing import canonical_sha256, file_sha256
from unituner.artifacts.tensors import save_tensors, weight_artifact_sha256
from unituner.checkpoints.restore import restore_checkpoint
from unituner.checkpoints.schema import build_resume_identity
from unituner.config.training import TrainingConfig
from unituner.data.cache.representation import RepaTargets, representation_dataset_sha256
from unituner.data.datasets.batching import DatasetBatches, EpochBatches
from unituner.data.datasets.restoration import AlignedImagePair, RestorationDataset
from unituner.data.media.image import discover_images, index_conditions
from unituner.data.media.restoration import read_restoration_pair
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
from unituner.models.minimax_h3.config.model import MiniMaxH3Config
from unituner.models.minimax_h3.image import load_image_transformer, save_image_transformer
from unituner.models.minimax_h3.image_cache import load_image_manifest, read_image_record, validate_image_entry
from unituner.precision import FLOAT_DTYPES
from unituner.training.contracts import RunState, StepResult
from unituner.training.engine import run_updates
from unituner.training.integrations.minimax_h3.image import MiniMaxH3ImageBinding, MiniMaxH3ImageFlowRequest
from unituner.training.integrations.minimax_h3.image_previews import MiniMaxH3ImagePreviews
from unituner.training.integrations.minimax_h3.restoration import MiniMaxH3ImageRestoration
from unituner.training.losses.wavelet import WaveletLoss
from unituner.training.objectives import align_steps, install_objectives
from unituner.training.optimizers.adamw import create_adamw
from unituner.training.output import publish_latest_weights, weights_path
from unituner.training.parameters import ParameterGroup, ParameterInstallation
from unituner.training.publishing import publish_adapter
from unituner.training.session import TrainingSession


def _restoration_pairs(config: TrainingConfig, manifests: dict) -> RestorationDataset:
    """Index LQ/HQ pairs against the exact high-image and text cache identities."""
    from unituner.data.production.minimax_h3_image import image_source_digest

    pairs: list[AlignedImagePair] = []
    digests = []
    for name, data in config.dataset.groups.items():
        high = discover_images(
            Path(data.source), caption_extension=None if data.prompt is not None else data.caption_extension
        )
        low = (
            None
            if data.degradation is not None and data.degradation.enabled
            else index_conditions(high, Path(data.low_quality_source))
        )
        references = tuple(index_conditions(high, Path(directory)) for directory in data.reference_sources)
        manifest = manifests[name]
        if len(high) != len(manifest.entries):
            raise ValueError("H3-Image restoration cache has a different number of sources")
        cache = Path(data.cache_directory)
        for path, entry in zip(high, manifest.entries, strict=True):
            caption_path = path.with_suffix(data.caption_extension or ".txt")
            caption = (
                data.prompt
                if data.prompt is not None
                else (caption_path.read_text(encoding="utf-8") if caption_path.is_file() else "")
            )
            ref_paths = tuple(index[path.stem] for index in references)
            identity = validate_image_entry(cache, entry)
            if identity.source_sha256 != image_source_digest(path, caption, ref_paths):
                raise ValueError("H3-Image restoration source, caption or reference changed; rebuild cache")
            pair = AlignedImagePair(None if low is None else low[path.stem], path, entry, cache, data.degradation)
            pairs.extend((pair,) * data.repeats)
            digests.append(
                {"group": name, "cache": entry.sha256, "low": None if low is None else file_sha256(low[path.stem])}
            )
    return RestorationDataset(tuple(pairs), canonical_sha256(digests), read_restoration_pair)


@runtime_scope
def train_minimax_h3_image(config: TrainingConfig, *, device: torch.device, resume: Path | None = None) -> RunState:
    """Use the shared optimizer/checkpoint engine for native image PDD weights and adapters."""
    config = resolve_training_precision(config)
    if config.model.family != ModelFamily.MINIMAX_H3_IMAGE or config.model.pdd_config is None:
        raise ValueError("H3-Image training requires its model family and PDD schedule")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("H3-Image training requires CPU/CUDA")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    execution, adapter = config.execution, config.train.adapter
    validate_training_execution(execution, device=device, adapter=adapter is not None)
    manifests = {
        name: load_image_manifest(
            Path(data.cache_directory), validation_workers=config.train.dataset_validation_workers
        )
        for name, data in config.dataset.groups.items()
    }
    restoration = config.method.kind == "restoration"
    dataset = _restoration_pairs(config, manifests) if restoration else None
    plans = {}
    for name, data in config.dataset.groups.items():
        if data.batch_size != 1 or data.kind != ("restoration" if restoration else "image"):
            raise ValueError("H3-Image training requires batch-one image/restoration caches")
        values = (
            (pair for pair in dataset.pairs if pair.cache_directory == Path(data.cache_directory))
            if dataset is not None
            else manifests[name].entries * data.repeats
        )
        plans[name] = EpochBatches({(0,): tuple(values)}, batch_size=1, seed=config.train.seed)
    plan = DatasetBatches(plans, seed=config.train.seed)
    representation_targets = None if config.method.repa is None else RepaTargets(config.method.repa)
    if representation_targets is not None:
        for manifest in manifests.values():
            representation_targets.validate_sources(manifest.entries)
    weight_path, schedule = Path(config.model.weights), Path(config.model.pdd_config)
    heads = None if config.model.pdd_heads is None else Path(config.model.pdd_heads)
    artifacts = {
        "transformer": weight_artifact_sha256(weight_path),
        "schedule": file_sha256(schedule),
        "heads": None if heads is None else weight_artifact_sha256(heads),
    }
    if restoration:
        options = config.method.restoration
        artifacts["vae"] = weight_artifact_sha256(Path(config.model.vae))
        if options.dino_weights is not None:
            artifacts["dino"] = weight_artifact_sha256(Path(options.dino_weights))
    dataset_digest = (
        dataset.digest
        if dataset is not None
        else canonical_sha256(
            [
                {"group": name, "manifest": file_sha256(Path(data.cache_directory) / "manifest.json")}
                for name, data in config.dataset.groups.items()
            ]
        )
    )
    identity = build_resume_identity(
        config,
        dataset_manifest_sha256=representation_dataset_sha256(dataset_digest, representation_targets),
        model_artifact_sha256=canonical_sha256(artifacts),
    )
    torch.manual_seed(config.train.seed)
    load_device = (
        torch.device("cpu") if execution.quantization != "none" or execution.offloads_frozen_weights else device
    )
    model = load_image_transformer(
        weight_path,
        config=config.model.image_topology or MiniMaxH3Config(),
        pdd_config=schedule,
        pdd_heads=heads,
        device=load_device,
        dtype=training_base_dtype(execution, adapter=adapter is not None),
    )
    if model.config.text_dim != 5120:
        raise ValueError("H3-Image training requires 5120-wide cached text")
    prepare_base_weights(model, execution, family=config.model.family.value, device=device)
    if execution.residency == "resident" and load_device != device:
        model.to(device)
    binding = MiniMaxH3ImageBinding(
        model,
        wavelet=None
        if config.method.wavelet is None or restoration
        else WaveletLoss(config.method.wavelet, device=device),
    )
    previews = None if config.preview is None else MiniMaxH3ImagePreviews(config, dataset)
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
        if execution.offloads_frozen_weights:
            session.own(FrozenModelPlacement(model, binding.regions, device=device))
        method = MiniMaxH3ImageRestoration(config, binding, device, session) if restoration else None
        groups = installation.plan.groups + (
            (ParameterGroup("encoder", tuple(method.encoder.parameters())),)
            if method is not None and method.encoder is not None
            else ()
        )
        if alignment is not None and alignment.projection is not None:
            groups += (ParameterGroup("repa", tuple(alignment.projection.parameters())),)
        optimizer = create_adamw(
            groups, lr=config.train.learning_rate, weight_decay=config.train.weight_decay, config=config.train.optimizer
        )
        optimizers = {"generator": optimizer} if restoration else optimizer
        modules = {"model": model, "adapter": installation.branches}
        if alignment is not None and alignment.projection is not None:
            modules["repa"] = alignment.projection
        if method is not None:
            modules.update(method.modules)
            if method.discriminator is not None:
                optimizers["discriminator"] = create_adamw(
                    (ParameterGroup("heads", tuple(method.discriminator.heads.parameters())),),
                    lr=config.train.learning_rate,
                    weight_decay=config.train.weight_decay,
                    config=config.train.optimizer,
                )
        record_context(config, installation.plan.target_names, identity)
        state = RunState()
        if resume is not None:
            cursor = restore_checkpoint(
                resume,
                modules,
                optimizers,
                identity,
                cuda_devices=(device.index or 0,) if device.type == "cuda" else (),
            )
            state = RunState(**cursor.model_dump())
            if (
                state.microbatches != state.updates * config.train.accumulation_steps
                or state.samples != plan.samples_before(state.microbatches)
                or state.updates > config.train.max_updates
            ):
                raise ValueError("H3-Image checkpoint cursor differs from the data plan")
        elif method is not None and method.options.calibrate:
            method.calibrate(config, dataset)
        session.own(ExecutionInstallation(binding.regions, execution, device=device))
        consumed = state.microbatches

        def raw_steps() -> Iterator[StepResult]:
            nonlocal consumed
            for _ in range(config.train.accumulation_steps):
                name, (item,) = plan.batch(consumed)
                data = config.dataset.groups[name]
                entry = item.caption if restoration else item
                record = read_image_record(Path(data.cache_directory), entry, device=device)
                if method is not None:
                    low, high = dataset.read(
                        item, data.resolution, config.train.seed + consumed, data.horizontal_flip_probability
                    )
                    consumed += 1
                    yield from method.steps(low.unsqueeze(0).to(device), high.unsqueeze(0).to(device), record)
                    del record, low, high
                else:
                    clean = record.latents.unsqueeze(0)
                    if alignment is not None:
                        alignment.set_targets((entry,), clean)
                    noise = torch.randn_like(clean)
                    if config.train.timestep_sampling == "uniform":
                        sigma = torch.rand((1,), device=device).clamp_min(1e-6)
                    else:
                        grid = model.pdd_sigmas[:-1]
                        sigma = torch.tensor([grid[int(torch.randint(len(grid), (1,), device=device))]], device=device)
                    request = MiniMaxH3ImageFlowRequest(
                        clean,
                        noise,
                        sigma,
                        record.text.unsqueeze(0),
                        record.token_tags.unsqueeze(0),
                        tuple(value.unsqueeze(0) for value in record.references),
                    )
                    consumed += 1
                    yield binding.step(request)
                    del record, clean, noise, request

        def write_weights(update: int) -> None:
            metadata = publication_metadata(config, identity, update)
            if method is not None:
                metadata["unituner.restoration.sigma"] = repr(method.sigma)
                metadata["unituner.restoration.posterior"] = method.posterior
                if method.encoder is not None:
                    save_tensors(
                        method.encoder.state_dict(),
                        weights_path(config, update=update, encoder=True),
                        dtype=FLOAT_DTYPES[config.output.save_precision],
                        metadata={
                            **metadata,
                            "unituner.encoder.targets": json.dumps(method.encoder.names),
                            "unituner.encoder.rank": str(method.options.encoder_rank),
                        },
                    )
            if adapter is None:
                save_image_transformer(
                    model,
                    weights_path(config, update=update),
                    metadata=metadata,
                    dtype=FLOAT_DTYPES[config.output.save_precision],
                )
            else:
                publish_adapter(
                    weights_path(config, update=update),
                    branches=installation.branches,
                    target_names=installation.plan.target_names,
                    adapter=adapter,
                    base_model="minimax_h3_image",
                    module_identifier="networks.lora_minimax_h3_image",
                    dtype=FLOAT_DTYPES[config.output.save_precision],
                    namespace="diffusion_model",
                    adapter_format=config.output.adapter_format,
                    family="minimax-h3-image",
                    metadata=metadata,
                )
            publish_latest_weights(config, update)

        run_updates(
            config,
            lambda: align_steps(raw_steps, alignment),
            optimizer=optimizers,
            modules=modules,
            state=state,
            identity=identity,
            device=device,
            weights_writer=write_weights,
            preview_writer=None if previews is None else lambda update: previews(update, model, binding, method),
        )
        if not weights_path(config).is_file():
            write_weights(state.updates)
    return state
