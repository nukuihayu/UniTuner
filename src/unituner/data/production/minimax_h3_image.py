"""Produce independent H3-Image text, target and ordered edit-reference caches."""

import hashlib
from pathlib import Path

import torch

from unituner.artifacts.hashing import canonical_sha256, file_sha256
from unituner.artifacts.tensors import weight_artifact_sha256
from unituner.config.cache import CacheConfig
from unituner.data.cache.feature_encoding import RepaCacheWriter
from unituner.data.cache.schema import CacheIdentity
from unituner.data.media.buckets import ImageBuckets
from unituner.data.media.image import discover_images, index_conditions, read_image
from unituner.models.components.text_artifacts import text_encoding_identity
from unituner.models.minimax_h3.image_cache import (
    MiniMaxH3ImageManifest,
    MiniMaxH3ImageRecord,
    save_image_manifest,
    save_image_record,
)
from unituner.models.minimax_h3.image_encoders import load_image_encoders
from unituner.models.minimax_h3.image_inference import encode_image_condition
from unituner.models.text_assets import tokenizer_sha256
from unituner.progress import progress


def image_source_digest(path: Path, caption: str, references: tuple[Path, ...] = ()) -> str:
    """Pin effective caption and all ordered input file contents for restoration indexing."""
    return canonical_sha256(
        {
            "image": file_sha256(path),
            "caption_utf8": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
            "references": [file_sha256(reference) for reference in references],
        }
    )


@torch.no_grad()
def produce_minimax_h3_image(
    config: CacheConfig, *, device: str, repa: RepaCacheWriter | None, tokenizer_digests: dict[str, str]
) -> int:
    model = config.model
    settings = model.minimax_h3
    if model.vae is None or model.text_encoder is None or settings is None:
        raise ValueError("H3-Image cache requires VAE, text encoder and model.minimax_h3.text_config")
    vae_device = torch.device(device)
    digest = canonical_sha256(
        {
            "runtime": config.runtime.model_dump(mode="json"),
            "tokenizers": tokenizer_digests,
            "vae": weight_artifact_sha256(Path(model.vae)),
            "text": weight_artifact_sha256(Path(model.text_encoder)),
            "processor": tokenizer_sha256(Path(model.text.tokenizer.source or "")),
            "text_config": file_sha256(Path(settings.text_config) / "config.json"),
            "text_source_format": model.text_source_format or "auto",
            "vae_topology": None if settings.video_topology is None else settings.video_topology.model_dump(),
            **text_encoding_identity(model.text),
        }
    )
    vae, text = load_image_encoders(model, device=vae_device)
    total = 0
    for data in config.dataset.groups.values():
        if data.kind not in ("image", "restoration"):
            raise ValueError("H3-Image cache requires image or restoration data")
        source, destination = Path(data.source), Path(data.cache_directory)
        buckets = ImageBuckets(
            data.resolution,
            alignment=32,
            enabled=data.enable_bucket and data.kind == "image",
            no_upscale=data.bucket_no_upscale,
            caption_extension=data.caption_extension,
            prompt=data.prompt,
        )
        files = discover_images(source, caption_extension=buckets.required_caption_extension)
        indices = tuple(index_conditions(files, Path(directory)) for directory in data.reference_sources)
        entries = []
        for path in progress(files, "[cache] Encoding H3-Image"):
            resolution = buckets.for_image(path)
            caption = buckets.caption(path)
            references = tuple(index[path.stem] for index in indices)
            pixels = read_image(path, resolution, rgb_before_resize=True).unsqueeze(0)
            text_rows, tags, conditions, _ = encode_image_condition(
                vae,
                text,
                caption,
                width=resolution[0],
                height=resolution[1],
                images=references,
                model_device=vae_device,
            )
            latents = vae.encode(pixels.to(vae_device).unsqueeze(2))[0].float().cpu()
            record = MiniMaxH3ImageRecord(
                latents,
                text_rows[0].float().cpu(),
                tags[0].long().cpu(),
                tuple(reference[0].float().cpu() for reference in conditions),
            )
            identity = CacheIdentity(
                source_sha256=image_source_digest(path, caption, references),
                encoder_sha256=digest,
                transform_sha256=canonical_sha256(
                    {
                        "resolution": resolution,
                        "bucket": data.enable_bucket,
                        "no_upscale": data.bucket_no_upscale,
                        "image_condition": "comfy-short-edge-1024-2048/picture-label/2",
                    }
                ),
                layout=f"minimax-h3-image/1/C1HW+LD+tags+R{len(references)}",
            )
            entry = save_image_record(destination, identity, record)
            if repa is not None:
                repa.publish_normalized(entry, pixels[0].unsqueeze(1))
            entries.append(entry)
        save_image_manifest(destination, MiniMaxH3ImageManifest(entries=tuple(entries)))
        total += len(entries)
    return total
