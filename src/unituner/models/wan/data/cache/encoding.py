"""Publish Wan video or independent-frame mean caches with native text and I2V conditions."""

import hashlib
from pathlib import Path
from typing import Literal

import av
import cv2
import numpy as np
import torch
from ftfy import __version__ as ftfy_version
from PIL import __version__ as pillow_version
from pydantic import JsonValue
from transformers import __version__ as transformers_version

from unituner.artifacts.hashing import canonical_sha256
from unituner.config.datasets import VideoDatasetConfig
from unituner.data.cache.feature_encoding import RepaCacheWriter
from unituner.data.cache.schema import CacheIdentity
from unituner.data.cache.storage import CacheEntry
from unituner.data.media.buckets import ImageBuckets
from unituner.data.media.image import discover_images, read_image
from unituner.data.media.video import (
    VideoWindow,
    discover_videos,
    read_video,
    select_video_windows,
    video_source_sha256,
    video_source_size,
)
from unituner.models.components.text_loading import text_execution, validate_cache_text
from unituner.models.qwen_image.vae import QwenImageVAE
from unituner.models.wan.conditioning import encode_latent_conditions
from unituner.models.wan.config.datasets import WanDatasetConfig
from unituner.models.wan.data.cache.storage import (
    WanCacheFormat,
    WanCacheManifest,
    WanCacheRecord,
    load_wan_manifest,
    save_wan_manifest,
    save_wan_record,
)
from unituner.models.wan.single_frame import encode_single_frame_conditions
from unituner.models.wan.text.encoder import WanTextEncoder
from unituner.models.wan.vision.encoder import WanImageEncoder
from unituner.progress import progress


def index_wan_sources(source: Path, *, kind: Literal["image", "video"]) -> tuple[Path, ...]:
    """Discover one complete ordered source and reject ambiguous caption/control stems."""
    paths = discover_images(source) if kind == "image" else discover_videos(source)
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Wan cache requires unique target stems")
    return paths


@torch.no_grad()
def cache_wan(
    source: Path,
    destination: Path,
    vae: QwenImageVAE,
    text: WanTextEncoder,
    *,
    kind: Literal["image", "video"],
    version: Literal["2.1", "2.2"],
    variant: Literal["t2v", "i2v"],
    resolution: tuple[int, int],
    buckets: ImageBuckets | None = None,
    encoder_sha256: str,
    data: WanDatasetConfig,
    control_source: str | None = None,
    video: VideoDatasetConfig | None = None,
    image: WanImageEncoder | None = None,
    repa: RepaCacheWriter | None = None,
) -> WanCacheManifest:
    """Cache finite native FP32 CFHW means, LD text and native I2V conditions.

    Read RGB uint8 before resize, preserve native cumulative fractional FPS or
    explicit timestamp grids, then select deterministic windows. Image sources
    encode one frame. I2V conditions encode first-frame plus normalized zero
    padding; Wan2.1 additionally extracts257 penultimate CLIP tokens. Controls
    follow the same frame grid, truncating or repeating the last decoded frame
    to target extent before windowing. No RCM or posterior sampling is used.

    Explicit single-frame image data binds one/two clean sources to temporal
    coordinates. Encode each frame with an independent causal history and
    preserve image-coordinate binding when sorting. Store the target position
    and per-clean-image CLIP features separately from regular video records.

    All encoders must be frozen eval FP32. Caller owns residency; encoding
    disables ambient autocast and leaves RNG/modes/input storage untouched.
    Content identities pin source/caption/control bytes, window indices,
    encoder artifacts, numeric runtime and transform choices. Preflight reused
    records and atomically publish the complete manifest last.
    """
    if min(resolution) < 16 or any(size % 16 for size in resolution):
        raise ValueError("Wan cache resolution must be positive16-aligned")
    if (image is not None) != (version == "2.1" and variant == "i2v"):
        raise ValueError("Wan CLIP image encoder is required exactly for2.1 I2V")
    if kind not in ("image", "video") or (kind == "image" and video is not None):
        raise ValueError("Wan video policy requires video data")
    single_frame = data.single_frame
    if single_frame is not None and control_source is not None:
        raise ValueError("Wan single_frame and control_source are mutually exclusive")
    if single_frame is not None and (kind != "image" or variant != "i2v"):
        raise ValueError("Wan single_frame requires I2V image data")
    policy = video or VideoDatasetConfig()
    if any((count - 1) % 4 for count in (*policy.target_frames, policy.max_frames)):
        raise ValueError("Wan target_frames/max_frames must follow1+4k")
    validate_cache_text(text.model)
    roots = (vae, text.model) if image is None else (vae, text.model, image.model)
    if any(module.training for root in roots for module in root.modules()) or any(
        parameter.requires_grad or parameter.dtype != torch.float32
        for root in roots
        if root is not text.model
        for parameter in root.parameters()
    ):
        raise ValueError("Wan cache encoders must be frozen eval FP32")
    if image is not None and (image.model.config.image_size != 224 or image.model.config.patch_size != 14):
        raise ValueError("Wan cache CLIP requires224/14 geometry with257tokens")
    targets = index_wan_sources(source, kind=kind)
    if buckets is not None and buckets.required_caption_extension is not None:
        targets = tuple(path for path in targets if path.with_suffix(buckets.required_caption_extension).is_file())
        if not targets:
            raise ValueError("Wan source contains no media matching caption_extension")
    controls: dict[str, Path] = {}
    if control_source is not None:
        paths = index_wan_sources(Path(control_source), kind=kind)
        controls = {path.stem: path for path in paths}
        if controls.keys() != {path.stem for path in targets}:
            raise ValueError("Wan controls require complete identical target stems")
    clean_maps: list[dict[str, Path]] = []
    if single_frame is not None:
        for directory in single_frame.clean_sources:
            clean = {path.stem: path for path in index_wan_sources(Path(directory), kind="image")}
            if clean.keys() != {path.stem for path in targets}:
                raise ValueError("Wan clean_sources require complete identical target stems")
            clean_maps.append(clean)
    geometry = WanCacheFormat(
        version=version,
        variant=variant,
        text_dim=text.model.config.hidden_size,
        image_dim=None if image is None else image.model.config.hidden_size,
        control=bool(controls),
        frame_layout="video" if single_frame is None else "single-frame",
        clean_count=len(clean_maps),
    )
    devices = {str(next(root.parameters()).device) for root in roots}
    device_names: list[JsonValue] = []
    device_names.extend(sorted(devices))
    transform = canonical_sha256(
        {
            **({"text_execution": text_execution(text.model)} if text_execution(text.model) else {}),
            "schema": "wan/regular-mean-text-image-control-1",
            "format": geometry.model_dump(mode="json"),
            "resolution_width_height": list(resolution),
            "kind": kind,
            "video_policy": None if kind == "image" else policy.model_dump(mode="json"),
            "single_frame": None if single_frame is None else single_frame.model_dump(mode="json"),
            "pixels": "RGB-before-resize/native-Lanczos-half-up-center-crop/no-EXIF;FP32uint8/127.5-1",
            "posterior": "per-row-mean/(raw-mean)*reciprocal_std/FP32",
            "control": "same-frame-grid/truncate-or-repeat-last/before-window",
            "vae": vae.config.model_dump(mode="json"),
            "text": text.model.config.model_dump(mode="json"),
            "max_text_length": text.max_length,
            "image": None if image is None else image.model.config.model_dump(mode="json"),
            "torch": torch.__version__,
            "av": av.__version__,
            "ffmpeg": {name: ".".join(map(str, version)) for name, version in av.library_versions.items()},
            "numpy": np.__version__,
            "pillow": pillow_version,
            "opencv": cv2.__version__,
            "transformers": transformers_version,
            "ftfy": ftfy_version,
            "devices": device_names,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "sdpa": {
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
                "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            },
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_devices": {
                name: {
                    "name": torch.cuda.get_device_properties(torch.device(name)).name,
                    "capability": list(torch.cuda.get_device_capability(torch.device(name))),
                }
                for name in sorted(devices)
                if torch.device(name).type == "cuda"
            },
        }
    )
    previous = load_wan_manifest(destination) if (destination / "manifest.json").exists() else None
    reusable = {} if previous is None else {entry.file: entry for entry in previous.entries}
    entries: list[CacheEntry] = []
    device = next(vae.parameters()).device
    for path in progress(targets, "[cache] Wan source encoding"):
        if buckets is not None:
            resolution = buckets.for_image(path) if kind == "image" else buckets.select(video_source_size(path))
        prompt = (buckets or ImageBuckets(resolution)).caption(path)
        source_hash = video_source_sha256(path)
        control_path = controls.get(path.stem)
        control_hash = None if control_path is None else video_source_sha256(control_path)
        clean_paths = tuple(clean[path.stem] for clean in clean_maps)
        clean_hashes: list[JsonValue] = [video_source_sha256(path) for path in clean_paths]
        if kind == "image":
            pixels = read_image(path, resolution, rgb_before_resize=True).unsqueeze(1)
            control_pixels = (
                None
                if control_path is None
                else read_image(control_path, resolution, rgb_before_resize=True).unsqueeze(1)
            )
            windows, source_indices = (VideoWindow(0, 1),), (0,)
        else:
            end = max(policy.target_frames) if policy.extraction == "head" else None
            decoded = read_video(path, resolution, policy, end_frame=end)
            pixels = torch.from_numpy(decoded.pixels).permute(3, 0, 1, 2).contiguous().float() / 127.5 - 1
            source_indices = decoded.source_indices
            windows = select_video_windows(pixels.shape[1], policy)
            control_pixels = None
            if control_path is not None:
                control_values = read_video(control_path, resolution, policy, end_frame=end).pixels
                if len(control_values) < len(decoded.pixels):
                    control_values = np.concatenate(
                        (
                            control_values,
                            np.repeat(control_values[-1:], len(decoded.pixels) - len(control_values), axis=0),
                        )
                    )
                control_pixels = (
                    torch.from_numpy(control_values[: len(decoded.pixels)]).permute(3, 0, 1, 2).contiguous().float()
                    / 127.5
                    - 1
                )
        caption = None
        for window in windows:
            identity = CacheIdentity(
                source_sha256=canonical_sha256(
                    {
                        "target": source_hash,
                        "control": control_hash,
                        "clean": clean_hashes,
                        "caption_utf8": hashlib.sha256(prompt.encode()).hexdigest(),
                        "window": [window.start, window.count],
                        "source_indices": list(source_indices[window.start : window.start + window.count]),
                    }
                ),
                encoder_sha256=encoder_sha256,
                transform_sha256=canonical_sha256({"policy": transform, "bucket": list(resolution)}),
                layout=geometry.layout,
            )
            repa_pixels = pixels[:, window.start : window.start + window.count]
            if repa is not None and single_frame is not None:
                # Every independent slot is supervised by the native flow caller.
                slots = {single_frame.target_index: repa_pixels[:, 0]}
                slots.update(
                    {
                        index: read_image(path, resolution, rgb_before_resize=True)
                        for index, path in zip(single_frame.clean_indices, clean_paths, strict=True)
                    }
                )
                repa_pixels = torch.stack([slots[index] for index in sorted(slots)], dim=1)
            name = identity.digest() + ".safetensors"
            if name in reusable:
                entries.append(reusable[name])
                if repa is not None:
                    repa.publish_normalized(entries[-1], repa_pixels)
                continue
            if caption is None:
                caption = text.encode([prompt])[0].float()
            crop = pixels[:, window.start : window.start + window.count].unsqueeze(0).to(device)
            control_crop = (
                None
                if control_pixels is None
                else control_pixels[:, window.start : window.start + window.count].unsqueeze(0).to(device)
            )
            if single_frame is not None:
                clean_pixels = (
                    torch.stack([read_image(path, resolution, rgb_before_resize=True) for path in clean_paths], dim=1)
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.autocast(device.type, enabled=False):
                    single_conditions = encode_single_frame_conditions(
                        vae,
                        crop,
                        clean_pixels,
                        clean_indices=single_frame.clean_indices,
                        target_index=single_frame.target_index,
                    )
                layout = single_conditions.layout
                features = None if image is None else image.encode((clean_pixels[0, :, layout.clean_order],))
                record = WanCacheRecord(
                    single_conditions.latents[0],
                    caption,
                    single_conditions.image_condition[0],
                    features,
                    frame_indices=layout.frame_indices,
                    target_position=layout.target_position,
                )
            else:
                with torch.autocast(device.type, enabled=False):
                    conditions = encode_latent_conditions(
                        vae, crop, image_conditioning=variant == "i2v", control_pixels=control_crop
                    )
                features = None if image is None else image.encode((crop[0, :, :1],))[0]
                record = WanCacheRecord(
                    conditions.latents[0],
                    caption,
                    None if conditions.image_condition is None else conditions.image_condition[0],
                    features,
                    None if conditions.control_latents is None else conditions.control_latents[0],
                )
            entry = save_wan_record(destination, identity, record, manifest=geometry)
            reusable[name] = entry
            entries.append(entry)
            if repa is not None:
                repa.publish_normalized(entry, repa_pixels)
    manifest = WanCacheManifest(**geometry.model_dump(), entries=tuple(entries))
    save_wan_manifest(manifest, destination)
    return manifest
