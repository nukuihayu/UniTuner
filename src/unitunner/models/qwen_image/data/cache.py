"""Produce native Qwen image/layer caches with explicit content and reference identities."""

import hashlib
from pathlib import Path

import torch
from PIL import Image
from transformers import __version__ as transformers_version

from unitunner.artifacts.hashing import canonical_sha256, file_sha256
from unitunner.config.dataset_inputs import DatasetInputConfig
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
from unitunner.data.media.image import discover_images, index_conditions, read_image, resize_pixels
from unitunner.models.components.text_loading import text_execution, validate_cache_text
from unitunner.models.qwen_image.data.media import reference_resolution
from unitunner.models.qwen_image.text.encoder import QwenImageTextEncoder
from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.progress import progress


@torch.no_grad()
def cache_qwen_image(
    source: Path,
    destination: Path,
    vae: QwenImageVAE,
    text: QwenImageTextEncoder,
    *,
    resolution: tuple[int, int],
    buckets: ImageBuckets | None = None,
    encoder_sha256: str,
    data: DatasetInputConfig,
    repa: RepaCacheWriter | None = None,
) -> LatentManifest:
    """Publish ordered FP32 CLHW targets, LD text and C1HW edit posterior means.

    Source images are composites for layered, foreground directories retain
    alpha and independent histories. Targets resize to the explicit(width,height).
    Controls resize first at VAE geometry, discard alpha, then resize that RGB
    result to384² VL buckets, following native cache caller order. Original edit
    text uses only the first reference; later edits retain all or none. Cache
    encoding and text are FP32 without generation's BF16 boundary rounding.
    Hash all source/caption/reference/layer contents and numerical policy, never
    path/mtime alone. Existing records are immutable; publish manifest last.
    Caller owns frozen eval FP32 encoders, modes and residency. No RNG changes.
    """
    variant = text.variant
    layered = variant == "layered"
    data.validate_for("qwen-image", variant)
    if min(resolution) < 16 or any(size % 16 for size in resolution):
        raise ValueError("Qwen cache resolution must be at least16 and divisible by16")
    if vae.config.input_channels != (4 if layered else 3):
        raise ValueError("Qwen cache vae RGB/RGBA channels differ from text variant")
    if layered != bool(data.layer_sources) or (variant in ("original", "layered") and data.reference_sources):
        raise ValueError("Qwen layered cache requires foregrounds; original/layered cannot consume edit references")
    if variant == "edit" and not data.reference_sources:
        raise ValueError("Original Qwen edit cache requires references")
    validate_cache_text(text.model)
    parameter, text_parameter = next(vae.parameters()), next(text.model.parameters())
    if any(module.training for root in (vae, text.model) for module in root.modules()) or any(
        parameter.dtype != torch.float32 or parameter.requires_grad
        for root in (vae,)
        for parameter in root.parameters()
    ):
        raise ValueError("Qwen cache encoders must be frozen eval FP32")
    files = discover_images(source, caption_extension=None if buckets is None else buckets.required_caption_extension)
    if len({path.stem for path in files}) != len(files):
        raise ValueError("Qwen cache targets require unique file stems")
    references = tuple(index_conditions(files, Path(path)) for path in data.reference_sources)
    layers = tuple(index_conditions(files, Path(path)) for path in data.layer_sources)
    for path in files:
        with Image.open(path) as image:
            size = image.size
        for layer in layers:
            with Image.open(layer[path.stem]) as foreground:
                if "A" not in foreground.getbands() or foreground.size != size:
                    raise ValueError("Qwen foregrounds require alpha and original dimensions matching the composite")
    transform = canonical_sha256(
        {
            **({"text_execution": text_execution(text.model)} if text_execution(text.model) else {}),
            "schema": "qwen-image/image-layer-text-1",
            "variant": variant,
            "resolution_width_height": list(resolution),
            "reference_bucket_policy": {
                "geometry": "independent-source-buckets-v1",
                "no_upscale": False if buckets is None else buckets.no_upscale,
                "enabled": False if buckets is None else buckets.enabled,
            },
            "reference_geometry": "independent-source-buckets/lanczos-up-area-down/half-up-center-crop",
            "reference_count": len(references),
            "foreground_count": len(layers),
            "pixels": "native-lanczos-up/area-down/half-up/center-crop/alpha-before-resize/127.5-1",
            "references": "vae-bucket-then-RGB-then-VL384-bucket/native-ordered",
            "posterior": "mean/FP32/independent-layers",
            "latents_mean": list(vae.config.latents_mean),
            "latents_std": list(vae.config.latents_std),
            "text": "native34-or64-prefix-drop/unpadded/FP32",
            "torch": torch.__version__,
            "transformers": transformers_version,
            "vae_device": str(parameter.device),
            "text_device": str(text_parameter.device),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_devices": {
                str(device): {
                    "name": torch.cuda.get_device_properties(device).name,
                    "capability": list(torch.cuda.get_device_capability(device)),
                }
                for device in {parameter.device, text_parameter.device}
                if device.type == "cuda"
            },
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
    )
    reference_buckets = buckets or ImageBuckets(resolution, alignment=16, enabled=False)
    previous = load_latent_manifest(destination) if (destination / "manifest.json").exists() else None
    reusable = {} if previous is None else {entry.file: entry for entry in previous.entries}
    entries: list[CacheEntry] = []
    for path in progress(files, "[cache] Qwen Image encoding"):
        resolution = resolution if buckets is None else buckets.for_image(path)
        prompt = (buckets or ImageBuckets(resolution)).caption(path)
        identity = CacheIdentity(
            source_sha256=canonical_sha256(
                {
                    "target": file_sha256(path),
                    "caption_utf8": hashlib.sha256(prompt.encode()).hexdigest(),
                    "references": [file_sha256(index[path.stem]) for index in references],
                    "foregrounds": [file_sha256(index[path.stem]) for index in layers],
                }
            ),
            encoder_sha256=encoder_sha256,
            transform_sha256=canonical_sha256({"policy": transform, "bucket": list(resolution)}),
            layout=f"qwen-image/{variant}/CLHW+LD+R{len(references)}",
        )
        filename = identity.digest() + ".safetensors"
        pixels = torch.stack(
            [
                read_image(item, resolution, channels=vae.config.input_channels)
                for item in (path, *(index[path.stem] for index in layers))
            ],
            dim=1,
        ).unsqueeze(0)
        if filename in reusable:
            entries.append(reusable[filename])
            if repa is not None:
                repa.publish_normalized(entries[-1], pixels[0])
            continue
        with torch.autocast(parameter.device.type, enabled=False):
            latents = vae.encode_layers(pixels.to(parameter.device))[0]
            control_latents, vision = [], []
            for index in references:
                control_path = index[path.stem]
                size = reference_buckets.for_image(control_path)
                control = read_image(control_path, size, channels=4)[:3]
                control_latents.append(vae.encode(control[None, :, None].to(parameter.device))[0])
                values = ((control + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
                vision.append(resize_pixels(values, reference_resolution(*size, area=384**2)))
        caption = text.encode_prompt(prompt, references=vision if variant.startswith("edit") else None)
        entries.append(
            save_latent_record(
                destination, identity, latents, caption.features[0].float(), references=tuple(control_latents)
            )
        )
        if repa is not None:
            repa.publish_normalized(entries[-1], pixels[0])
    manifest = LatentManifest.model_validate({"family": "qwen-image", "variant": variant, "entries": tuple(entries)})
    save_latent_manifest(manifest, destination)
    return manifest
