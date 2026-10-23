"""Assemble native minimax_h3 cache resources and dataset traversal."""

from pathlib import Path

import torch

from unitunner.artifacts.hashing import canonical_sha256, file_sha256
from unitunner.artifacts.tensors import weight_artifact_sha256
from unitunner.config.cache import CacheConfig
from unitunner.config.datasets import CacheDatasetConfig
from unitunner.data.cache.feature_encoding import RepaCacheWriter
from unitunner.data.media.buckets import ImageBuckets
from unitunner.models.components.text_artifacts import text_encoding_identity
from unitunner.models.minimax_h3.data.cache.encoding import build_cache_encoders, cache_minimax_h3
from unitunner.models.minimax_h3.data.cache.inputs import MiniMaxH3CachePolicy
from unitunner.models.minimax_h3.data.cache.storage import MiniMaxH3CacheFormat
from unitunner.models.minimax_h3.data.cache.traversal import MiniMaxH3TraversalPolicy, traverse_minimax_h3
from unitunner.models.minimax_h3.data.sources import MiniMaxH3Record, load_records
from unitunner.models.minimax_h3.text.loading import load_text_encoder
from unitunner.models.minimax_h3.vae import load_audio_vae, load_video_vae
from unitunner.models.text_assets import tokenizer_sha256
from unitunner.progress import status


def produce_minimax_h3(
    config: CacheConfig, *, device: str, repa: RepaCacheWriter | None, tokenizer_digests: dict[str, str]
) -> int:
    """Load frozen native H3 encoders and atomically publish JSONL target caches."""
    model = config.model
    encoders_config = model.minimax_h3
    if encoders_config is None or model.vae is None or model.text_encoder is None:
        raise ValueError("MiniMaxH3 cache requires model.vae, model.text_encoder and model.minimax_h3 artifacts")
    video_path, text_path = Path(model.vae), Path(model.text_encoder)
    selected = torch.device(device)
    text_device = selected if encoders_config.text_device is None else torch.device(encoders_config.text_device)
    if selected.type not in ("cpu", "cuda") or text_device.type not in ("cpu", "cuda"):
        raise ValueError("MiniMaxH3 cache supports CPU/CUDA devices")
    sources: list[
        tuple[
            CacheDatasetConfig,
            MiniMaxH3CacheFormat,
            MiniMaxH3CachePolicy,
            MiniMaxH3TraversalPolicy,
            tuple[MiniMaxH3Record, ...],
        ]
    ] = []
    for data in config.dataset.groups.values():
        if data.source is None or data.cache_directory is None or model.variant not in ("t2va", "fl2va", "ref2va"):
            raise ValueError("MiniMaxH3 cache requires a JSONL source, cache_directory and t2va/fl2va/ref2va variant")
        policy = (
            MiniMaxH3CachePolicy()
            if data.minimax_h3 is None
            else MiniMaxH3CachePolicy.model_validate(data.minimax_h3.model_dump())
        )
        traversal = (
            MiniMaxH3TraversalPolicy(resolution=data.resolution)
            if data.video is None
            else MiniMaxH3TraversalPolicy(resolution=data.resolution, video=data.video)
        )
        format = MiniMaxH3CacheFormat(
            task=model.variant, frame_layout="single-frame" if data.kind == "image" else "video"
        )
        records = load_records(Path(data.source), format.task, one_frame=data.kind == "image", prompt=data.prompt)
        if any(record.teacher_caption is not None for record in records):
            raise ValueError("MiniMaxH3 teacher-caption cache production is not implemented")
        if data.kind == "image" and format.task == "fl2va" and any(not record.controls for record in records):
            raise ValueError("MiniMaxH3 single-frame FL2VA requires controls for every target")
        directory = Path(data.cache_directory).resolve()
        if Path(data.source).resolve() == directory / "manifest.json":
            raise ValueError("MiniMaxH3 output manifest must not overwrite the input source")
        if not policy.allow_experimental_duration and data.video is not None and data.video.extraction != "full":
            if any(not 120 <= count <= 360 for count in data.video.target_frames):
                raise ValueError("MiniMaxH3 target duration must be5-15 seconds unless explicitly allowed")

        sources.append((data, format, policy, traversal, records))

    def artifact_digest() -> str:
        """Hash actual local files plus explicit topology/source-format choices."""
        return canonical_sha256(
            {
                "runtime": config.runtime.model_dump(mode="json"),
                "tokenizers": dict(tokenizer_digests),
                "video": weight_artifact_sha256(video_path),
                "audio": weight_artifact_sha256(Path(encoders_config.audio_vae)),
                "text": weight_artifact_sha256(text_path),
                "text_config": file_sha256(Path(encoders_config.text_config) / "config.json"),
                "processor": tokenizer_sha256(Path(model.text.tokenizer.source or "")),
                "video_topology": None
                if encoders_config.video_topology is None
                else encoders_config.video_topology.model_dump(),
                "audio_topology": None
                if encoders_config.audio_topology is None
                else encoders_config.audio_topology.model_dump(),
                "text_source_format": config.model.text_source_format or "native",
                **text_encoding_identity(model.text),
            }
        )

    status("[cache] Hashing model artifacts and validating inputs")
    encoder_digest = artifact_digest()
    status("[cache] Loading video vae")
    video = (
        load_video_vae(Path(model.vae), config=encoders_config.video_topology, include_decoder=False, device=selected)
        .eval()
        .requires_grad_(False)
    )
    status("[cache] Loaded video vae")
    status("[cache] Loading audio vae")
    audio = (
        load_audio_vae(
            Path(encoders_config.audio_vae),
            config=encoders_config.audio_topology,
            include_decoder=False,
            device=selected,
        )
        .eval()
        .requires_grad_(False)
    )
    status("[cache] Loaded audio vae")
    status("[cache] Loading text encoder")
    text = load_text_encoder(
        Path(model.text_encoder),
        config_directory=Path(encoders_config.text_config),
        processor_directory=None,
        device=text_device,
        source_format=config.model.text_source_format or "native",
        text_config=config.model.text,
    )
    status("[cache] Loaded text encoder")
    encoders = build_cache_encoders(video, audio, text)

    def validate_artifacts() -> None:
        """Reject changed artifacts before publishing immutable records or the manifest."""
        if artifact_digest() != encoder_digest:
            raise ValueError("MiniMaxH3 encoder artifacts changed during loading or cache production")

    validate_artifacts()
    count = 0
    for data, format, policy, traversal, records in sources:
        manifest = cache_minimax_h3(
            traverse_minimax_h3(
                records,
                format=format,
                traversal=traversal,
                policy=policy,
                buckets=ImageBuckets(
                    data.resolution, alignment=32, enabled=data.enable_bucket, no_upscale=data.bucket_no_upscale
                ),
            ),
            Path(data.cache_directory).resolve(),
            encoders,
            format=format,
            policy=policy,
            repa=repa,
            encoder_sha256=encoder_digest,
            validate_encoder_artifacts=validate_artifacts,
        )
        count += len(manifest.entries)
    return count
