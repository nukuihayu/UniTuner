"""Encode native image/text caches with explicit content and preprocessing identities."""

import hashlib
from contextlib import ExitStack
from pathlib import Path

import torch
from torch import Tensor
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
from unitunner.data.media.image import discover_images, index_conditions, read_image
from unitunner.models.components.conditioning import Caption
from unitunner.models.components.text_loading import text_execution, validate_cache_text
from unitunner.models.flux.conditioning import pack_fill_mask
from unitunner.models.flux.data.encoding import encode_latents
from unitunner.models.flux.data.media import read_fill_condition
from unitunner.models.flux.text.encoder import FluxTextEncoder
from unitunner.models.flux.vae import FluxVAE
from unitunner.models.flux2.text.encoder import Flux2TextEncoder
from unitunner.models.flux2.vae import Flux2VAE
from unitunner.models.flux3.text.encoder import Flux3TextEncoder
from unitunner.models.moss_image.text.encoder import MossImageTextEncoder
from unitunner.models.z_image.text.encoder import ZImageTextEncoder
from unitunner.models.z_image.vae import ZImageVAE
from unitunner.progress import progress


def _packed_mean(vae: Flux2VAE, pixels: Tensor) -> Tensor:
    """Encode one image into normalized packed CHW posterior means."""
    return vae.encode(pixels)[0]


def _cache_z_image_batch(
    items: list[tuple[Path, tuple[int, int], str, CacheIdentity]],
    destination: Path,
    vae: ZImageVAE,
    text: ZImageTextEncoder,
    reusable: dict[str, CacheEntry],
    repa: RepaCacheWriter | None,
) -> list[CacheEntry]:
    """Encode a stable batch, retaining per-image records and native caption masks."""
    sources = [identity.source_sha256 for _, _, _, identity in items]
    identities = [
        identity.model_copy(
            update={"transform_sha256": canonical_sha256({"item": identity.transform_sha256, "batch_sources": sources})}
        )
        for _, _, _, identity in items
    ]
    cached = [reusable.get(identity.digest() + ".safetensors") for identity in identities]
    if all(entry is not None for entry in cached):
        entries = [entry for entry in cached if entry is not None]
        if repa is not None:
            for entry, (path, resolution, _, _) in zip(entries, items, strict=True):
                repa.publish_normalized(entry, read_image(path, resolution).unsqueeze(1))
        return entries
    # Encode the complete original batch even on partial reuse: changing peers
    # can change kernel rounding, so retries must retain the same inputs/order.
    pixels = torch.stack([read_image(path, resolution) for path, resolution, _, _ in items]).to(
        next(vae.parameters()).device
    )
    with ExitStack() as stack:
        for device_type in {pixels.device.type, next(text.model.parameters()).device.type}:
            stack.enter_context(torch.autocast(device_type, enabled=False))
        latents = vae.normalize(vae.encode(pixels).mean)
        captions = text.encode([prompt for _, _, prompt, _ in items])
    entries = []
    for index, identity in enumerate(identities):
        features = captions.features[index]
        if captions.mask is not None:
            features = features[captions.mask[index]]
        entry = cached[index]
        if entry is None:
            entry = save_latent_record(destination, identity, latents[index], features.float())
        entries.append(entry)
        if repa is not None:
            repa.publish_normalized(entry, pixels[index].unsqueeze(1))
    return entries


@torch.no_grad()
def cache_images(
    source: Path,
    destination: Path,
    vae: ZImageVAE | Flux2VAE | FluxVAE,
    text: ZImageTextEncoder | MossImageTextEncoder | FluxTextEncoder | Flux2TextEncoder | Flux3TextEncoder,
    *,
    resolution: tuple[int, int],
    buckets: ImageBuckets | None = None,
    encoder_sha256: str,
    variant: str | None = None,
    data: DatasetInputConfig | None = None,
    version: str | None = None,
    seed: int = 42,
    repa: RepaCacheWriter | None = None,
    batch_size: int = 1,
) -> LatentManifest:
    """Encode ordered images and optional UTF-8 .txt sidecars into native caches.

    Use model-native normalization and text semantics; Flux preserves all T5
    padding and CLIP pooled features. FLUX3 retains encoder padding and ordered
    clean reference means rounded to BF16 then stored as FP32. Flux posterior
    uses image-content-seeded draws from the run seed; other families use posterior
    mean. Caller supplies
    owned frozen eval encoders and a
    digest covering both artifacts. Publish the manifest only after all records
    succeed. Old manifests survive failures; newly completed records are reusable.
    This eager FP32 producer does not alter global RNG or encoder modes.
    """
    moss, flux, flux3 = (
        isinstance(text, MossImageTextEncoder),
        isinstance(text, FluxTextEncoder),
        isinstance(text, Flux3TextEncoder),
    )
    flux2 = isinstance(text, Flux2TextEncoder)
    if isinstance(vae, Flux2VAE) != (moss or flux2 or flux3) or isinstance(vae, FluxVAE) != flux:
        raise ValueError("Cache vae/text families must match")
    family = "flux2" if flux2 else "flux3" if flux3 else "flux" if flux else "moss-image" if moss else "z-image"
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Cache batch_size must be a positive integer")
    if batch_size > 1 and (not isinstance(text, ZImageTextEncoder) or not isinstance(vae, ZImageVAE)):
        raise ValueError("Batched caching currently supports Z-Image only")
    if flux2:
        if variant not in ("base", "distilled") or version not in ("dev", "klein-4b", "klein-9b"):
            raise ValueError("FLUX.2 cache requires explicit version and base/distilled variant")
        if (
            not isinstance(text, Flux2TextEncoder)
            or version != text.version
            or (version == "dev" and variant != "distilled")
        ):
            raise ValueError("FLUX.2 cache version/variant differs from text or published topology")
    elif version is not None:
        raise ValueError("FLUX.2 reference/version cache policy requires FLUX.2 text")
    if flux3:
        if variant not in ("base", "distilled"):
            raise ValueError("FLUX3 cache requires explicit base/distilled variant")
    if flux:
        if variant not in ("dev", "kontext", "fill"):
            raise ValueError("Flux cache requires explicit dev/kontext/fill variant")
        if variant != "dev" and not (data and data.reference_sources):
            raise ValueError("Flux Kontext/Fill requires a reference source")
        if variant == "fill" and isinstance(vae, FluxVAE) and vae.config.spatial_factor != 8:
            raise ValueError("Flux Fill requires the native vae spatial factor eight")
    elif variant is not None and not (flux2 or flux3 or (moss and variant == "edit")):
        raise ValueError("Variant cache policy is supported only for Flux")
    data = DatasetInputConfig() if data is None else data
    data.validate_for(family, variant or "base")
    posterior = "sample" if flux else "mean"
    moss_edit = moss and variant == "edit"
    files = discover_images(source, caption_extension=None if buckets is None else buckets.required_caption_extension)
    reference_indices = (
        tuple(index_conditions(files, Path(directory)) for directory in data.reference_sources)
        if flux2 or flux3 or moss_edit
        else ()
    )
    conditions = index_conditions(files, Path(data.reference_sources[0])) if flux and data.reference_sources else {}
    masks = {} if data.mask_source is None else index_conditions(files, Path(data.mask_source))
    normalization = (
        {
            "packed": "channel-row-column/2x2",
            "normalization": "frozen-checkpoint-BN",
            "epsilon": vae.config.normalization_eps,
        }
        if isinstance(vae, Flux2VAE)
        else {"latent_scale": vae.config.scaling_factor, "latent_shift": vae.config.shift_factor}
    )
    if min(resolution) < 16 or any(value % 16 for value in resolution):
        raise ValueError("Image cache resolution must be at least 16 and divisible by 16")
    parameter = next(vae.parameters())
    text_models = (text.t5, text.clip) if isinstance(text, FluxTextEncoder) else (text.model,)
    text_parameters = tuple(next(model.parameters()) for model in text_models)
    for model in text_models:
        validate_cache_text(model)
    if any(module.training for model in (vae, *text_models) for module in model.modules()) or any(
        value.dtype != torch.float32 for value in (parameter,)
    ):
        raise ValueError("Cache encoders must be in eval mode and FP32")
    transform = canonical_sha256(
        {
            **(
                {"text_execution": [text_execution(model) for model in text_models]}
                if any(text_execution(model) for model in text_models)
                else {}
            ),
            "schema": f"{family}/image-text-1",
            **({"batch_size": batch_size} if batch_size > 1 else {}),
            "resolution_width_height": list(resolution),
            "resize": "lanczos-up/area-down/half-up/center-crop",
            "pixels": "RGB/drop-alpha/no-exif/float32/127.5-1",
            "posterior": posterior,
            "posterior_seed": seed if flux else None,
            "posterior_rng": "image-content-local-CPU-target-then-condition" if flux else None,
            "variant": variant,
            "version": version,
            "fill_mask": "nearest/bucket-aligned/white-repaint" if variant == "fill" else None,
            **({"fill_rgb": "drop-alpha-before-resize/shared-mask-geometry-v2"} if variant == "fill" else {}),
            **normalization,
            "text": "Mistral3/system/layers10-20-30/left512"
            if flux2 and version == "dev"
            else "Qwen3/thinking-disabled/layers9-18-27/retained512"
            if flux2
            else "Qwen3-VL/processor-template/eight-layers/retained-padding"
            if flux3
            else "T5/final/unmasked/512+CLIP/ftfy/pooled/77"
            if flux
            else "thinking-enabled/final/unpadded"
            if moss
            else "thinking-enabled/penultimate/right-padding",
            "max_length": 4096
            if isinstance(text, Flux3TextEncoder)
            else 512
            if isinstance(text, Flux2TextEncoder)
            else text.max_length,
            **(
                {
                    "references": "target-bucket/alpha-after-resize/no-exif/mean-BF16-roundtrip",
                    "reference_count": len(reference_indices),
                    "text_padding_multiple": 80,
                    "text_layers": [4, 8, 12, 16, 20, 24, 28, 32],
                    "text_output_dtype": "bfloat16",
                }
                if flux3
                else {}
            ),
            **(
                {
                    "references": "target-bucket/alpha-after-resize/no-exif/mean-BF16-roundtrip",
                    "reference_count": len(reference_indices),
                    "text_layers": [10, 20, 30] if version == "dev" else [9, 18, 27],
                    "text_output_dtype": "float32",
                }
                if flux2
                else {}
            ),
            **(
                {
                    "references": "target-bucket/mean-FP32",
                    "reference_count": len(reference_indices),
                    "edit_contract": "index_timestep_zero_text_start",
                }
                if moss_edit
                else {}
            ),
            "torch": torch.__version__,
            "transformers": transformers_version,
            "vae_device": str(parameter.device),
            "text_devices": [str(value.device) for value in text_parameters],
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
                for device in {parameter.device, *(value.device for value in text_parameters)}
                if device.type == "cuda"
            },
        }
    )
    previous = load_latent_manifest(destination) if (destination / "manifest.json").exists() else None
    reusable = {} if previous is None else {entry.file: entry for entry in previous.entries}
    entries: list[CacheEntry] = []
    pending: list[tuple[Path, tuple[int, int], str, CacheIdentity]] = []

    def flush() -> None:
        if pending:
            assert isinstance(vae, ZImageVAE) and isinstance(text, ZImageTextEncoder)
            entries.extend(_cache_z_image_batch(pending, destination, vae, text, reusable, repa))
            pending.clear()

    for path in progress(files, f"[cache] {family} images"):
        resolution = resolution if buckets is None else buckets.for_image(path)
        prompt = (buckets or ImageBuckets(resolution)).caption(path)
        source_digest = canonical_sha256(
            {
                "image": file_sha256(path),
                "caption_utf8": hashlib.sha256(prompt.encode()).hexdigest(),
                **({"condition": file_sha256(conditions[path.stem])} if conditions else {}),
                **({"mask": file_sha256(masks[path.stem])} if masks else {}),
                **(
                    {"references": [file_sha256(index[path.stem]) for index in reference_indices]}
                    if flux2 or flux3 or moss_edit
                    else {}
                ),
            }
        )
        identity = CacheIdentity(
            source_sha256=source_digest,
            encoder_sha256=encoder_sha256,
            transform_sha256=canonical_sha256({"policy": transform, "bucket": list(resolution)}),
            layout=f"flux2/{version}/{variant}/CHW+LD+R{len(reference_indices)}"
            if flux2
            else f"flux3/{variant}/CHW+LD+R{len(reference_indices)}"
            if flux3
            else f"flux/{variant}/CHW+LD+V"
            if flux
            else f"moss-image/edit/CHW+LD+R{len(reference_indices)}"
            if moss_edit
            else f"{family}/CHW+LD",
        )
        if batch_size > 1:
            if pending and pending[0][1] != resolution:
                flush()
            pending.append((path, resolution, prompt, identity))
            if len(pending) == batch_size:
                flush()
            continue
        filename = identity.digest() + ".safetensors"
        if filename in reusable:
            entries.append(reusable[filename])
            if repa is not None:
                repa.publish_normalized(entries[-1], read_image(path, resolution).unsqueeze(1))
            continue
        pixels = read_image(path, resolution).unsqueeze(0).to(parameter.device)
        with ExitStack() as stack:
            for device_type in {parameter.device.type, *(value.device.type for value in text_parameters)}:
                stack.enter_context(torch.autocast(device_type, enabled=False))
            pooled, control_latents, fill_mask = None, None, None
            reference_latents = ()
            if isinstance(vae, FluxVAE) and isinstance(text, FluxTextEncoder):
                image_seed = (int(file_sha256(path)[:16], 16) + seed) % (2**63)
                generator = torch.Generator(device="cpu").manual_seed(image_seed)

                latents = encode_latents(vae, pixels, generator, sample=posterior == "sample")
                if conditions:
                    if variant == "fill":
                        control_pixels, mask = read_fill_condition(
                            path, conditions[path.stem], masks.get(path.stem), resolution
                        )
                        fill_mask = pack_fill_mask(mask.unsqueeze(0))[0]
                    else:
                        control_pixels = read_image(conditions[path.stem], resolution)
                    control_latents = encode_latents(
                        vae,
                        control_pixels.unsqueeze(0).to(parameter.device),
                        generator,
                        sample=posterior == "sample",
                    )
                encoded = text.encode([prompt])
                features, pooled = encoded.captions[0].float(), encoded.pooled[0].float()
            else:
                if isinstance(vae, Flux2VAE):
                    latents = _packed_mean(vae, pixels)
                elif isinstance(vae, ZImageVAE):
                    latents = vae.normalize(vae.encode(pixels).mean)[0]
                else:
                    raise TypeError("Flux cache requires matched native text provider")
                if isinstance(text, Flux2TextEncoder | Flux3TextEncoder) and isinstance(vae, Flux2VAE):
                    reference_latents = tuple(
                        _packed_mean(vae, read_image(index[path.stem], resolution)[None].to(parameter.device))
                        .to(torch.bfloat16)
                        .float()
                        .cpu()
                        for index in reference_indices
                    )
                    caption = text.encode_prompt(prompt)
                else:
                    if moss_edit and isinstance(vae, Flux2VAE):
                        reference_latents = tuple(
                            _packed_mean(vae, read_image(index[path.stem], resolution)[None].to(parameter.device))
                            .float()
                            .cpu()
                            for index in reference_indices
                        )
                    caption = text.encode([prompt])
                if not isinstance(caption, Caption):
                    raise TypeError("Image text encoder must return a native caption")
                features = (
                    caption.features[0].float()
                    if caption.mask is None
                    else caption.features[0, caption.mask[0]].float()
                )
        entries.append(
            save_latent_record(
                destination,
                identity,
                latents,
                features,
                pooled=pooled,
                control_latents=control_latents,
                fill_mask=fill_mask,
                references=reference_latents,
            )
        )
        if repa is not None:
            repa.publish_normalized(entries[-1], pixels[0].unsqueeze(1))
    flush()
    manifest = LatentManifest.model_validate(
        {
            "family": family,
            "variant": variant if flux or flux2 or flux3 or moss_edit else None,
            "version": version,
            "entries": tuple(entries),
        }
    )
    save_latent_manifest(manifest, destination)
    return manifest
