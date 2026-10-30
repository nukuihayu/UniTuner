"""Assemble native wan cache resources and dataset traversal."""

from pathlib import Path

import torch

from unituner.artifacts.hashing import canonical_sha256
from unituner.artifacts.tensors import weight_artifact_sha256
from unituner.config.cache import CacheConfig
from unituner.data.cache.feature_encoding import RepaCacheWriter
from unituner.data.media.buckets import ImageBuckets
from unituner.models.components.text_artifacts import text_encoding_identity
from unituner.models.qwen_image.loading import load_qwen_image_vae
from unituner.models.wan.config.datasets import WanDatasetConfig
from unituner.models.wan.data.cache.encoding import cache_wan, index_wan_sources
from unituner.models.wan.text.encoder import load_wan_text
from unituner.models.wan.vision.encoder import WanImageEncoder
from unituner.models.wan.vision.loading import load_wan_clip
from unituner.progress import status


def produce_wan(
    config: CacheConfig, *, device: str, repa: RepaCacheWriter | None, tokenizer_digests: dict[str, str]
) -> int:
    """Assemble regular Wan FP32 mean/text/image cache owners from explicit native artifacts."""
    model = config.model
    if model.vae is None or model.text_encoder is None:
        raise ValueError("Wan cache requires model.vae and model.text_encoder")
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("Wan cache supports CPU/CUDA devices")
    if model.version not in ("2.1", "2.2") or model.variant not in ("t2v", "i2v"):
        raise ValueError("Wan cache requires explicit2.1/2.2 t2v/i2v configuration")
    for data in config.dataset.groups.values():
        if data.kind not in ("image", "video"):
            raise ValueError("Wan cache requires image/video data")
        if min(data.resolution) < 16 or any(size % 16 for size in data.resolution):
            raise ValueError("Wan cache resolution requires16-aligned dimensions")
        if data.source is None or data.cache_directory is None:
            raise ValueError("Wan cache requires one source and cache_directory")
        targets = index_wan_sources(Path(data.source), kind=data.kind)
        if data.control_source is not None:
            controls = index_wan_sources(Path(data.control_source), kind=data.kind)
            if {path.stem for path in targets} != {path.stem for path in controls}:
                raise ValueError("Wan controls require complete identical target stems")
        if data.wan is not None and data.wan.single_frame is not None:
            for directory in data.wan.single_frame.clean_sources:
                clean = index_wan_sources(Path(directory), kind="image")
                if {path.stem for path in targets} != {path.stem for path in clean}:
                    raise ValueError("Wan clean_sources require complete identical target stems")
    image_path = model.asset_path("image_encoder") if (model.version, model.variant) == ("2.1", "i2v") else None
    status("[cache] Hashing model artifacts and validating inputs (large weights may take a while)")
    encoder_digest = canonical_sha256(
        {
            "runtime": config.runtime.model_dump(mode="json"),
            "tokenizers": dict(tokenizer_digests),
            "vae": weight_artifact_sha256(Path(model.vae)),
            "text": weight_artifact_sha256(Path(model.text_encoder)),
            **text_encoding_identity(model.text),
            "image": None if image_path is None else weight_artifact_sha256(image_path),
        }
    )
    status("[cache] Loading qwen image vae")
    vae = load_qwen_image_vae(Path(model.vae), device=selected).eval().requires_grad_(False)
    status("[cache] Loaded qwen image vae")
    status("[cache] Loading wan text")
    text = load_wan_text(Path(model.text_encoder), device=selected, text_config=config.model.text)
    status("[cache] Loaded wan text")
    status("[cache] Loading wan clip")
    image = None if image_path is None else WanImageEncoder(load_wan_clip(image_path, device=selected))
    status("[cache] Loaded wan clip")
    count = 0
    for data in config.dataset.groups.values():
        if data.kind not in ("image", "video"):
            raise ValueError("Wan cache requires image/video data")
        manifest = cache_wan(
            Path(data.source),
            Path(data.cache_directory),
            vae,
            text,
            kind=data.kind,
            version=model.version,
            variant=model.variant,
            resolution=data.resolution,
            buckets=ImageBuckets(
                data.resolution,
                enabled=data.enable_bucket,
                no_upscale=data.bucket_no_upscale,
                caption_extension=data.caption_extension,
                prompt=data.prompt,
            ),
            repa=repa,
            encoder_sha256=encoder_digest,
            data=WanDatasetConfig() if data.wan is None else data.wan,
            control_source=data.control_source,
            video=data.video,
            image=image,
        )
        count += len(manifest.entries)
    return count
