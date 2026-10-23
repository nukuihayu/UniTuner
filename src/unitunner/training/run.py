"""Assemble typed training workflows at the optional Torch application boundary."""

from collections.abc import Callable
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import cast

import torch
from pydantic import JsonValue

from unitunner.artifacts.hashing import canonical_sha256
from unitunner.config.training import TrainingConfig
from unitunner.data.datasets.restoration import AlignedImagePair, RestorationDataset, index_restoration_pairs
from unitunner.execution.precision import resolve_training_precision
from unitunner.models.catalog import ModelFamily
from unitunner.progress import status
from unitunner.training.output import prepare_training_output
from unitunner.training.workflows.common.images import PreviewWriter, train
from unitunner.training.workflows.minimax_h3.run import train_minimax_h3
from unitunner.training.workflows.wan.run import train_wan


def run(config: TrainingConfig, *, device: str, resume: Path | None) -> int:
    """Assemble a typed local device and run the shared native training application."""
    config = resolve_training_precision(prepare_training_output(config, resume=resume))
    status(
        f"[train] Starting {config.model.family}/{config.method.kind} on {device}; precision={config.execution.precision}"
    )
    if config.model.family == ModelFamily.MINIMAX_H3:
        train_minimax_h3(config, device=torch.device(device), resume=resume)
        return 0
    if config.model.family == ModelFamily.WAN:
        video_preview = None
        if config.preview is not None:
            if any(find_spec(name) is None for name in ("PIL", "cv2", "av", "transformers")):
                raise ValueError("Wan previews require uv sync --extra media --extra text")
            video_preview = import_module("unitunner.training.integrations.wan.previews").VideoPreviews(config)
        train_wan(config, device=torch.device(device), resume=resume, preview_writer=video_preview)
        return 0
    restoration_data = None
    if config.method.kind == "restoration":
        status("[train] Indexing restoration pairs and validating caption caches")
        if find_spec("PIL") is None:
            raise ValueError("Image restoration training requires uv sync --extra media")
        discover = cast(Callable[..., tuple[Path, ...]], import_module("unitunner.data.media.image").discover_images)
        read = cast(
            Callable[[AlignedImagePair, tuple[int, int], int, float], tuple[torch.Tensor, torch.Tensor]],
            import_module("unitunner.data.media.restoration").read_restoration_pair,
        )
        if not callable(discover) or not callable(read):
            raise TypeError("Image boundary entries must be callable")
        sources = tuple(config.dataset.groups.values())
        caches = [Path(source.cache_directory).resolve() for source in sources]
        if len(set(caches)) != len(caches):
            raise ValueError("Distinct dataset sources require distinct resolved cache directories")
        pairs = []
        identities: list[JsonValue] = []
        for name, source in config.dataset.groups.items():
            if source.source is None or source.cache_directory is None:
                raise ValueError("Restoration requires source and cache_directory")
            indexed, digest = index_restoration_pairs(
                discover(
                    Path(source.source),
                    caption_extension=None if source.prompt is not None else source.caption_extension,
                ),
                None if source.low_quality_source is None else discover(Path(source.low_quality_source)),
                Path(source.cache_directory),
                degradation=source.degradation,
                family=config.model.family,
                caption_extension=source.caption_extension,
                prompt=source.prompt,
            )
            pairs.extend(indexed * source.repeats)
            identities.append({"group": name, "digest": digest, "repeats": source.repeats})
        restoration_data = RestorationDataset(tuple(pairs), canonical_sha256({"sources": identities}), read)
        status(f"[train] Indexed {len(pairs)} restoration pairs (including repeats)")
    preview_writer = None
    if config.preview is not None:
        required = ("PIL", "cv2", "transformers") if config.method.kind == "flow-matching" else ("PIL", "cv2")
        if any(find_spec(name) is None for name in required):
            raise ValueError("Image previews require media dependencies; flow prompts also require the text extra")
        factory = import_module("unitunner.training.previews.images").PreviewImages
        if not callable(factory):
            raise TypeError("Preview image provider must be callable")
        preview_writer = cast(PreviewWriter, factory(config, restoration_data))
    train(
        config,
        device=torch.device(device),
        resume=resume,
        restoration_data=restoration_data,
        preview_writer=preview_writer,
    )
    return 0
