"""Produce unified Qwen2.1 mean/text caches from shared alpha-preserving reference pixels."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import __version__ as transformers_version

from unitunner.artifacts.hashing import canonical_sha256, file_sha256
from unitunner.config.dataset_inputs import DatasetInputConfig
from unitunner.data.cache.feature_encoding import RepaCacheWriter
from unitunner.data.cache.schema import CacheIdentity
from unitunner.data.cache.storage import CacheEntry, LatentManifest, load_latent_manifest, save_latent_manifest
from unitunner.data.media.buckets import ImageBuckets
from unitunner.data.media.image import discover_images, index_conditions, resize_pixels
from unitunner.models.components.text_loading import text_execution, validate_cache_text
from unitunner.models.qwen_image.data.media import reference_resolution
from unitunner.models.qwen_image_2_1.data.cache.storage import QwenImage21CacheRecord, save_qwen_image_21_record
from unitunner.models.qwen_image_2_1.data.encoding import encode_images
from unitunner.models.qwen_image_2_1.text.encoder import QwenImage21TextEncoder
from unitunner.models.qwen_image_2_1.vae import QwenImage21VAE
from unitunner.progress import progress


@torch.no_grad()
def cache_qwen_image_21(
    source: Path,
    destination: Path,
    vae: QwenImage21VAE,
    text: QwenImage21TextEncoder,
    *,
    resolution: tuple[int, int],
    buckets: ImageBuckets | None = None,
    encoder_sha256: str,
    data: DatasetInputConfig,
    repa: RepaCacheWriter | None = None,
) -> LatentManifest:
    """Publish native FP32 64x1xHxW means, LD text and exact ordered reference metadata.

    Targets and references independently use the dataset bucket settings.
    Equal source dimensions produce matching resize and crop geometry.
    reference_resolution overrides the reference area budget.
    One RGBA byte array feeds both encoders and reference hashing: VAE retains alpha, vision alone
    composites white. No EXIF rotation or independent VLM resize occurs. Content
    identities include caption/source/reference contents, all numerical choices
    and supplied encoder artifact digest. Reuse validates the previous manifest;
    publish the new manifest last. Caller owns frozen eval FP32 encoders and
    residency; encoding leaves modes/RNG unchanged.
    """
    data.validate_for("qwen-image-2-1", "base")
    if min(resolution) < 32 or any(size % 32 for size in resolution):
        raise ValueError("Qwen Image2.1 cache resolution must be positive multiples of32")
    if vae.config.latent_channels != 64 or vae.config.spatial_factor != 16:
        raise ValueError("Qwen Image2.1 public cache requires native latent64 and spatial factor16")
    validate_cache_text(text.model)
    roots = (vae, text.model)
    if any(module.training for root in roots for module in root.modules()) or any(
        p.dtype != torch.float32 or p.requires_grad for root in roots[:-1] for p in root.parameters()
    ):
        raise ValueError("Qwen Image2.1 cache encoders must be frozen eval FP32")
    files = discover_images(source, caption_extension=None if buckets is None else buckets.required_caption_extension)
    if len({path.stem for path in files}) != len(files):
        raise ValueError("Qwen Image2.1 cache targets require unique file stems")
    references = tuple(index_conditions(files, Path(path)) for path in data.reference_sources)
    parameter, text_parameter = next(vae.parameters()), next(text.model.parameters())
    transform = canonical_sha256(
        {
            **({"text_execution": text_execution(text.model)} if text_execution(text.model) else {}),
            "schema": "qwen-image-2-1/unified-image-text-1",
            "resolution_width_height": list(resolution),
            "reference_bucket_policy": {
                "geometry": "independent-source-buckets-v1",
                "no_upscale": False if buckets is None else buckets.no_upscale,
                "enabled": False if buckets is None else buckets.enabled,
            },
            "reference_geometry": (
                "independent-aspect-bucket/lanczos-up-area-down/half-up-center-crop"
                if data.reference_resolution is not None
                else "independent-source-buckets/lanczos-up-area-down/half-up-center-crop"
            ),
            **(
                {"independent_reference_resolution": list(data.reference_resolution), "reference_alignment": 32}
                if data.reference_resolution is not None
                else {}
            ),
            "reference_count": len(references),
            "pixels": "RGBA-before-resize/native-Lanczos-up-area-down-half-up-center-crop/no-EXIF",
            "vision": "same-pixels/white-alpha-composite/no-resize/patch16-merge2",
            "posterior": "mean/FP32/direct-divide-std",
            "vae_config": vae.config.model_dump(mode="json"),
            "text": "Qwen3-VL/pre-final-RMSNorm/system-padding-vision-removed/ordered-slots",
            "torch": torch.__version__,
            "transformers": transformers_version,
            "vae_device": str(parameter.device),
            "text_device": str(text_parameter.device),
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
                for device in {parameter.device, text_parameter.device}
                if device.type == "cuda"
            },
        }
    )
    reference_buckets = buckets or ImageBuckets(resolution, alignment=32, enabled=False)
    previous = load_latent_manifest(destination) if (destination / "manifest.json").exists() else None
    reusable = {} if previous is None else {entry.file: entry for entry in previous.entries}
    entries: list[CacheEntry] = []
    for path in progress(files, "[cache] Qwen Image 2.1 encoding"):
        resolution = resolution if buckets is None else buckets.for_image(path)
        prompt = (buckets or ImageBuckets(resolution)).caption(path)
        identity = CacheIdentity(
            source_sha256=canonical_sha256(
                {
                    "target": file_sha256(path),
                    "caption_utf8": hashlib.sha256(prompt.encode()).hexdigest(),
                    "references": [file_sha256(group[path.stem]) for group in references],
                }
            ),
            encoder_sha256=encoder_sha256,
            transform_sha256=canonical_sha256({"policy": transform, "bucket": list(resolution)}),
            layout=f"qwen-image-2-1/C1HW+LD+R{len(references)}",
        )
        name = identity.digest() + ".safetensors"
        with Image.open(path) as image:
            target = resize_pixels(np.array(image.convert("RGBA"), copy=True), resolution)
        if name in reusable:
            entries.append(reusable[name])
            if repa is not None:
                repa.publish(entries[-1], torch.from_numpy(target).unsqueeze(0))
            continue
        controls = []
        for group in references:
            with Image.open(group[path.stem]) as image:
                rgba = np.array(image.convert("RGBA"), copy=True)
            control_size = reference_buckets.for_image(group[path.stem])
            if data.reference_resolution is not None:
                control_size = reference_resolution(
                    rgba.shape[1],
                    rgba.shape[0],
                    area=data.reference_resolution[0] * data.reference_resolution[1],
                    alignment=32,
                )
            controls.append(resize_pixels(rgba, control_size))
        # Vision validates every original reference grid before either encoder runs.
        features = text.encode([prompt], references=[controls])[0]
        latents = encode_images(vae, [target, *controls])
        hashes = []
        for image in controls:
            height, width = image.shape[:2]
            hashes.append(list(hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).digest()))
        record = QwenImage21CacheRecord(
            latents[0],
            features.features.float(),
            tuple(latents[1:]),
            torch.tensor(features.image_slots, dtype=torch.int64),
            torch.tensor(features.image_shapes, dtype=torch.int64).reshape(-1, 2),
            torch.tensor(hashes, dtype=torch.uint8).reshape(-1, 32),
        )
        entries.append(save_qwen_image_21_record(destination, identity, record))
        if repa is not None:
            repa.publish(entries[-1], torch.from_numpy(target).unsqueeze(0))
    manifest = LatentManifest(family="qwen-image-2-1", entries=tuple(entries))
    save_latent_manifest(manifest, destination)
    return manifest
