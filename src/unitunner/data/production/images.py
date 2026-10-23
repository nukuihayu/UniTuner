"""Validate image groups and assemble native image/text cache production."""

from pathlib import Path

import torch

from unitunner.artifacts.hashing import canonical_sha256
from unitunner.artifacts.tensors import weight_artifact_sha256
from unitunner.config.cache import CacheConfig
from unitunner.data.cache.feature_encoding import RepaCacheWriter
from unitunner.data.media.buckets import ImageBuckets
from unitunner.data.media.image import discover_images, index_conditions
from unitunner.data.production.image_encoders import load_image_encoders
from unitunner.models.catalog import ModelFamily
from unitunner.models.components.text_artifacts import text_artifact_sha256, text_encoding_identity
from unitunner.progress import status


def produce_images(
    config: CacheConfig, *, device: str, repa: RepaCacheWriter | None, tokenizer_digests: dict[str, str]
) -> int:
    """Load native FP32 VAEs and independently configured text encoders for caching."""
    if config.model.family not in {
        ModelFamily.Z_IMAGE,
        ModelFamily.MOSS_IMAGE,
        ModelFamily.FLUX,
        ModelFamily.FLUX3,
        ModelFamily.FLUX2,
        ModelFamily.QWEN_IMAGE,
        ModelFamily.QWEN_IMAGE_2_1,
        ModelFamily.KREA2,
    } or any(
        data.kind
        not in {
            "image",
            "restoration",
        }
        for data in config.dataset.groups.values()
    ):
        raise ValueError("Cache entry supports Z-Image, MOSS, Flux, FLUX.2, FLUX3 Qwen Image/2.1 and Krea2 datasets")
    vae_path, text_path = config.model.vae, config.model.text_encoder
    if vae_path is None or text_path is None:
        raise ValueError("Cache production requires model.vae and model.text_encoder")
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("Cache entry supports CPU and CUDA devices")
    sources = tuple(config.dataset.groups.values())
    if config.model.family == ModelFamily.QWEN_IMAGE_2_1:
        if any(min(data.resolution) < 32 or any(size % 32 for size in data.resolution) for data in sources):
            raise ValueError("Qwen Image2.1 cache requires32-aligned resolution")
    destinations = [Path(source.cache_directory).resolve() for source in sources]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Distinct dataset sources must not publish to the same resolved cache directory")
    secondary = config.model.asset_path("clip_text") if config.model.family == ModelFamily.FLUX else None
    if config.model.family == ModelFamily.FLUX and any(data.kind != "image" for data in sources):
        raise ValueError("Flux cache requires image data and the CLIP text_encoder bundle component")
    prepared = []
    for source in sources:
        if min(source.resolution) < 16 or any(value % 16 for value in source.resolution):
            raise ValueError("Image cache resolution must be at least 16 and divisible by 16")
        if source.source is None or source.cache_directory is None:
            raise ValueError("Image source requires source and cache_directory")
        buckets = ImageBuckets(
            source.resolution,
            alignment=32 if config.model.family == ModelFamily.QWEN_IMAGE_2_1 else 16,
            enabled=source.enable_bucket and source.kind != "restoration",
            no_upscale=source.bucket_no_upscale,
            caption_extension=source.caption_extension,
            prompt=source.prompt,
        )
        source.validate_for(config.model.family, config.model.variant)
        buckets.select(source.resolution)
        prepared.append((source, buckets))
        targets = discover_images(Path(source.source))
        for directory in (*source.reference_sources, *source.layer_sources):
            index_conditions(targets, Path(directory))
        if source.mask_source is not None:
            index_conditions(targets, Path(source.mask_source))
    status("[cache] Hashing model artifacts and validating inputs (large weights may take a while)")
    encoder_digest = canonical_sha256(
        {
            "runtime": config.runtime.model_dump(mode="json"),
            "tokenizers": dict(tokenizer_digests),
            "vae": weight_artifact_sha256(Path(vae_path)),
            "text": text_artifact_sha256(Path(text_path), config.model.family, version=config.model.version),
            **text_encoding_identity(config.model.text),
            **({"secondary_text": text_artifact_sha256(Path(secondary), "flux-clip")} if secondary is not None else {}),
        }
    )
    vae, text = load_image_encoders(config.model, device=selected)
    count = 0
    for source, buckets in prepared:
        if config.model.family == ModelFamily.KREA2:
            from unitunner.models.krea2.data.cache import cache_krea2

            manifest = cache_krea2(
                Path(source.source),
                Path(source.cache_directory),
                vae,
                text,
                resolution=source.resolution,
                buckets=buckets,
                repa=repa,
                encoder_sha256=encoder_digest,
            )
        elif config.model.family == ModelFamily.QWEN_IMAGE_2_1:
            from unitunner.models.qwen_image_2_1.data.cache.encoding import cache_qwen_image_21

            manifest = cache_qwen_image_21(
                Path(source.source),
                Path(source.cache_directory),
                vae,
                text,
                resolution=source.resolution,
                buckets=buckets,
                repa=repa,
                encoder_sha256=encoder_digest,
                data=source,
            )
        elif config.model.family == ModelFamily.QWEN_IMAGE:
            from unitunner.models.qwen_image.data.cache import cache_qwen_image

            manifest = cache_qwen_image(
                Path(source.source),
                Path(source.cache_directory),
                vae,
                text,
                resolution=source.resolution,
                buckets=buckets,
                repa=repa,
                encoder_sha256=encoder_digest,
                data=source,
            )
        else:
            from unitunner.data.cache.images import cache_images

            manifest = cache_images(
                Path(source.source),
                Path(source.cache_directory),
                vae,
                text,
                resolution=source.resolution,
                buckets=buckets,
                repa=repa,
                encoder_sha256=encoder_digest,
                variant=config.model.variant
                if config.model.family in (ModelFamily.FLUX, ModelFamily.FLUX2, ModelFamily.FLUX3)
                or (config.model.family == ModelFamily.MOSS_IMAGE and config.model.variant == "edit")
                else None,
                data=source,
                seed=config.seed,
                version=config.model.version if config.model.version in ("dev", "klein-4b", "klein-9b") else None,
                batch_size=config.cache.batch_size,
            )
        count += len(manifest.entries)
    return count
