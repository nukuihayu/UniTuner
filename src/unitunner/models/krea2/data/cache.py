"""Publish unified Krea2 image posterior means and compact selected-layer text caches."""

import hashlib
from pathlib import Path

import torch
from transformers import __version__ as transformers_version

from unitunner.artifacts.hashing import canonical_sha256, file_sha256
from unitunner.data.cache.feature_encoding import RepaCacheWriter
from unitunner.data.cache.schema import CacheIdentity
from unitunner.data.cache.storage import (
    CacheEntry,
    LatentManifest,
    load_latent_manifest,
    save_latent_manifest,
    save_latent_record,
)
from unitunner.data.media.buckets import ImageBuckets
from unitunner.data.media.image import discover_images, read_image
from unitunner.models.components.text_loading import text_execution, validate_cache_text
from unitunner.models.krea2.text.encoder import Krea2TextEncoder
from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.progress import progress


@torch.no_grad()
def cache_krea2(
    source: Path,
    destination: Path,
    vae: QwenImageVAE,
    text: Krea2TextEncoder,
    *,
    resolution: tuple[int, int],
    buckets: ImageBuckets | None = None,
    encoder_sha256: str,
    repa: RepaCacheWriter | None = None,
) -> LatentManifest:
    """Cache native FP32 16x1xHxW means and SLD captions without global RNG changes.

    Read RGB in [-1,1], resize/crop to (width,height)16-aligned buckets and encode
    the shared Qwen VAE's posterior mode with native reciprocal-std scaling.
    Text preserves ordered selected states and compacts valid tokens, including
    the suffix after padding. UTF-8 .txt sidecars are optional. Hash source,
    caption, complete encoder artifact digest and numerical transform policy.
    Records are immutable; preflight reusable files and publish the manifest last.
    Caller owns frozen eval FP32 encoders and residency. No reference/RCM inputs
    enter this plain text-to-image cache.
    """
    if min(resolution) < 16 or any(size % 16 for size in resolution):
        raise ValueError("Krea2 cache resolution requires positive16-aligned buckets")
    if vae.config.input_channels != 3 or vae.config.latent_channels != 16 or vae.config.spatial_factor != 8:
        raise ValueError("Krea2 cache requires shared RGB Qwen vae with latent16 and factor8")
    validate_cache_text(text.model)
    roots = (vae, text.model)
    if any(module.training for root in roots for module in root.modules()) or any(
        parameter.dtype != torch.float32 or parameter.requires_grad
        for root in roots[:-1]
        for parameter in root.parameters()
    ):
        raise ValueError("Krea2 cache encoders must be frozen eval FP32")
    vae_device, text_device = next(vae.parameters()).device, next(text.model.parameters()).device
    transform = canonical_sha256(
        {
            **({"text_execution": text_execution(text.model)} if text_execution(text.model) else {}),
            "schema": "krea2/unified-image-text-1",
            "resolution_width_height": list(resolution),
            "pixels": "RGB/native-Lanczos-up-area-down-half-up-center-crop/no-EXIF",
            "posterior": "mode/(raw-mean)*reciprocal_std/FP32",
            "vae_config": vae.config.model_dump(mode="json"),
            "text_config": text.config.model_dump(mode="json"),
            "text": "Qwen3-VL/decoder-inputs/prefix34/suffix5-after-padding/mask-positions/compact",
            "torch": torch.__version__,
            "transformers": transformers_version,
            "vae_device": str(vae_device),
            "text_device": str(text_device),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_devices": {
                str(device): {
                    "name": torch.cuda.get_device_properties(device).name,
                    "capability": list(torch.cuda.get_device_capability(device)),
                }
                for device in {vae_device, text_device}
                if device.type == "cuda"
            },
        }
    )
    previous = load_latent_manifest(destination) if (destination / "manifest.json").exists() else None
    reusable = {} if previous is None else {entry.file: entry for entry in previous.entries}
    entries: list[CacheEntry] = []
    for path in progress(
        discover_images(source, caption_extension=None if buckets is None else buckets.required_caption_extension),
        "[cache] Krea2 encoding",
    ):
        resolution = resolution if buckets is None else buckets.for_image(path)
        prompt = (buckets or ImageBuckets(resolution)).caption(path)
        identity = CacheIdentity(
            source_sha256=canonical_sha256(
                {"image": file_sha256(path), "caption_utf8": hashlib.sha256(prompt.encode()).hexdigest()}
            ),
            encoder_sha256=encoder_sha256,
            transform_sha256=canonical_sha256({"policy": transform, "bucket": list(resolution)}),
            layout="krea2/C1HW+SLD",
        )
        name = identity.digest() + ".safetensors"
        if name in reusable:
            entries.append(reusable[name])
            if repa is not None:
                repa.publish_normalized(entries[-1], read_image(path, resolution).unsqueeze(1))
            continue
        pixels = read_image(path, resolution).to(vae_device)
        latents = vae.encode(pixels.unsqueeze(0).unsqueeze(2))[0]
        caption = text.encode([prompt])[0].float()
        entries.append(save_latent_record(destination, identity, latents, caption))
        if repa is not None:
            repa.publish_normalized(entries[-1], pixels.unsqueeze(1))
    manifest = LatentManifest(family="krea2", entries=tuple(entries))
    save_latent_manifest(manifest, destination)
    return manifest
