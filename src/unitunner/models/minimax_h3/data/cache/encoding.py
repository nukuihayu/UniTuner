"""Encode decoded H3 windows and ordered references into complete immutable AV caches."""

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch
from PIL import __version__ as pillow_version
from torch import Tensor
from transformers import __version__ as transformers_version

from unitunner.artifacts.hashing import canonical_sha256, file_sha256
from unitunner.data.cache.feature_encoding import RepaCacheWriter
from unitunner.data.cache.schema import CacheIdentity
from unitunner.data.cache.storage import CacheEntry
from unitunner.models.components.text_loading import text_execution, validate_cache_text
from unitunner.models.minimax_h3.conditioning import encode_audio, encode_condition_video
from unitunner.models.minimax_h3.data.cache.inputs import MiniMaxH3CachePolicy, MiniMaxH3DecodedItem, build_cache_item
from unitunner.models.minimax_h3.data.cache.storage import (
    MiniMaxH3CacheFormat,
    MiniMaxH3CacheManifest,
    MiniMaxH3CacheRecord,
    build_record_layout,
    load_minimax_h3_manifest,
    read_minimax_h3_record,
    save_minimax_h3_manifest,
    save_minimax_h3_record,
)
from unitunner.models.minimax_h3.data.cache.video import encode_target_video
from unitunner.models.minimax_h3.data.encoding import encode_reference_conditions
from unitunner.models.minimax_h3.data.media.inputs import prepare_pixels
from unitunner.models.minimax_h3.layout import one_frame_condition_roles
from unitunner.models.minimax_h3.text.encoder import MiniMaxH3TextEncoder, MiniMaxH3TextFeatures
from unitunner.models.minimax_h3.text.presentation import (
    MiniMaxH3Presentation,
    MiniMaxH3TextReference,
    MiniMaxH3TextVisual,
    build_presentation,
)
from unitunner.models.minimax_h3.vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
from unitunner.progress import progress


@dataclass(frozen=True, slots=True)
class MiniMaxH3CacheEncoders:
    """Borrow typed numerical encoders and their declared execution identity.

    Video callables accept normalized B3THW pixels and return B24FHW. Target
    arguments are pixels, cache_seed, canonical_item_key. Audio accepts stereo2L
    and returns B32ST means. Text returns native pre-norm features/tags. Each
    callable preserves inputs, model modes and global RNG; callers own residency.
    """

    target_video: Callable[[Tensor, int, str], Tensor]
    condition_video: Callable[[Tensor], Tensor]
    audio: Callable[[Tensor], Tensor]
    text: Callable[[MiniMaxH3Presentation], MiniMaxH3TextFeatures]
    execution_sha256: str


def _numeric_runtime_sha256() -> str:
    """Pin actual invocation backend state, including contexts entered after encoder construction."""
    return canonical_sha256(
        {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "numpy": np.__version__,
            "av": av.__version__,
            "ffmpeg": {name: list(version) for name, version in av.library_versions.items()},
            "pillow": pillow_version,
            "precision": "FP32/no-autocast",
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "sdpa": {
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
                "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            },
        }
    )


def build_cache_encoders(
    video: MiniMaxH3VideoVAE, audio: MiniMaxH3AudioVAE, text: MiniMaxH3TextEncoder
) -> MiniMaxH3CacheEncoders:
    """Borrow frozen eval FP32 native owners without loading weights or installing hooks.

    VAEs may omit their unused decoders. Encoders may reside on different
    devices; every numerical invocation disables ambient autocast locally.
    Model topology, resident devices and precision/backend choices participate
    in execution identity, independently of the caller's artifact content hash.
    """
    validate_cache_text(text.model)
    roots = (video, audio, text.model)
    if any(module.training for root in roots for module in root.modules()) or any(
        value.requires_grad or value.dtype != torch.float32 for root in roots[:-1] for value in root.parameters()
    ):
        raise ValueError("MiniMaxH3 cache encoders must be frozen eval FP32")
    if video.vae_ratio != 16 or video.vae_ratio_t != 4 or video.clip_length != 17 or video.token_drop != 3:
        raise ValueError("MiniMaxH3 cache requires native16x spatial/4x temporal vae and17-frame chunks/drop3")
    video_device = next(video.parameters()).device
    audio_device = next(audio.parameters()).device

    def target(pixels: Tensor, seed: int, key: str) -> Tensor:
        with torch.autocast(video_device.type, enabled=False):
            return encode_target_video(video, pixels.to(video_device), cache_seed=seed, canonical_item_key=key)

    def condition(pixels: Tensor) -> Tensor:
        with torch.autocast(video_device.type, enabled=False):
            return encode_condition_video(video, pixels.to(video_device))

    def waveform(samples: Tensor) -> Tensor:
        with torch.autocast(audio_device.type, enabled=False):
            return encode_audio(audio, samples)

    execution = canonical_sha256(
        {
            **({"text_execution": text_execution(text.model)} if text_execution(text.model) else {}),
            "video": video.config.model_dump(mode="json"),
            "audio": audio.config.model_dump(mode="json"),
            "text": text.model.config.to_dict(),
            "devices": [str(next(root.parameters()).device) for root in roots],
            "numeric_runtime": _numeric_runtime_sha256(),
        }
    )
    return MiniMaxH3CacheEncoders(target, condition, waveform, text.encode, execution)


def _canonical_item_key(decoded: MiniMaxH3DecodedItem, format: MiniMaxH3CacheFormat) -> str:
    if format.frame_layout == "single-frame":
        return f"{decoded.record.target_path if decoded.item_key is None else decoded.item_key}#1f"
    return f"{decoded.record.target_path}#{decoded.crop_start_frame}:{len(decoded.frames)}"


def _one_sample(value: Tensor, dimensions: int) -> Tensor:
    if value.ndim != dimensions + 1 or value.shape[0] != 1:
        raise ValueError("MiniMaxH3 cache encoder must return exactly one native sample")
    return value[0]


@torch.no_grad()
def encode_cache_record(
    decoded: MiniMaxH3DecodedItem,
    encoders: MiniMaxH3CacheEncoders,
    *,
    format: MiniMaxH3CacheFormat,
    policy: MiniMaxH3CachePolicy,
) -> MiniMaxH3CacheRecord:
    """Encode one target, ordered references and student text without loading resources.

    Reuse accepted native posterior/mean encoders. Reference visuals are decoded
    once for both VAE and text: video VAE uses24Hz, text samples2Hz with
    timestamps0,.5,...; video references retain their own synchronized audio
    duration, while standalone audio uses a non-exact target-duration window.
    Single-frame references use the released360-frame cap. No peak rescaling,
    caller mutation, global RNG or RCM masks are introduced.
    """
    item = build_cache_item(decoded, format, policy)
    video = _one_sample(
        encoders.target_video(prepare_pixels(decoded.frames), policy.cache_seed, _canonical_item_key(decoded, format)),
        4,
    )
    audio = _one_sample(encoders.audio(decoded.waveform), 3)
    visual_conditions: list[Tensor] = []
    audio_conditions: list[Tensor] = []
    text_references: list[MiniMaxH3TextReference] = []
    anchors: dict[str, MiniMaxH3TextVisual] = {}
    if format.task == "fl2va":
        if format.frame_layout == "single-frame":
            controls = tuple(control.unsqueeze(0) for control in decoded.control_frames)
            roles = one_frame_condition_roles(len(controls))
        else:
            controls, roles = (decoded.frames[:1], decoded.frames[-1:]), ("first", "last")
        for role, frames in zip(roles, controls, strict=True):
            visual_conditions.append(_one_sample(encoders.condition_video(prepare_pixels(frames)), 4))
            anchors[role] = MiniMaxH3TextVisual(frames)
    if format.task == "ref2va":
        conditions = encode_reference_conditions(
            decoded.record.references,
            frame_cap=360 if format.frame_layout == "single-frame" else item.frame_count,
            resolution=(decoded.frames.shape[2], decoded.frames.shape[1]),
            standalone_audio_samples=decoded.waveform.shape[1],
            condition_video=encoders.condition_video,
            audio=encoders.audio,
            audio_decode=policy.audio_decode,
            audio_pad_tolerance=policy.audio_pad_tolerance,
            audio_max_repair_samples=policy.audio_max_repair_samples,
        )
        visual_conditions.extend(conditions.visual_conditions)
        audio_conditions.extend(conditions.audio_conditions)
        text_references.extend(conditions.text_references)
    presentation = build_presentation(decoded.record.caption, format.task, anchors=anchors, references=text_references)
    text = encoders.text(presentation)
    record = MiniMaxH3CacheRecord(
        item,
        video,
        audio,
        text.hidden_states.float(),
        text.token_tags,
        tuple(visual_conditions),
        tuple(audio_conditions),
    )
    build_record_layout(record, format=format)
    return record


def _tensor_sha256(value: Tensor) -> str:
    """Hash exact logical dtype/shape/content without constructing a Python byte string."""
    digest = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}\0".encode())
    data = value.detach().contiguous().cpu().view(torch.uint8).numpy()
    digest.update(memoryview(data))
    return digest.hexdigest()


def _source_files(decoded: MiniMaxH3DecodedItem) -> dict[str, str]:
    paths = {decoded.record.target_path}
    paths.update(control.path for control in decoded.record.controls)
    for reference in decoded.record.references:
        paths.add(reference.path)
        if reference.audio is not None:
            paths.add(reference.audio.path)
    if decoded.audio_source is not None:
        paths.add(decoded.audio_source.path)
    return {str(path.resolve()): file_sha256(path) for path in sorted(paths)}


@torch.no_grad()
def cache_minimax_h3(
    inputs: Iterable[MiniMaxH3DecodedItem],
    destination: Path,
    encoders: MiniMaxH3CacheEncoders,
    *,
    format: MiniMaxH3CacheFormat,
    policy: MiniMaxH3CachePolicy,
    encoder_sha256: str,
    validate_encoder_artifacts: Callable[[], None] | None = None,
    repa: RepaCacheWriter | None = None,
) -> MiniMaxH3CacheManifest:
    """Publish ordered complete AV records from shared decoded-media traversal.

    Caller supplies the digest of all VAE/text/processor artifacts and owns
    frozen encoder resources. Hash raw source files, captions, exact decoded
    target/control/waveform values, clock choices, crop/temporal coordinates,
    posterior item key, policies and execution. Labels are diagnostic only.
    Preflight previous files before reuse; a failure preserves the old manifest
    and completed immutable records. Files/inputs must remain immutable during
    the traversal; detect source changes across encoding before publication.
    An optional caller-owned artifact validator runs before publishing each
    encoded record and the final manifest, so changed weights cannot leave a
    reusable record under an obsolete encoder digest. Validation errors propagate.
    """
    previous = load_minimax_h3_manifest(destination) if (destination / "manifest.json").exists() else None
    if previous is not None and MiniMaxH3CacheFormat.model_validate(previous.model_dump(exclude={"entries"})) != format:
        raise ValueError("MiniMaxH3 destination manifest has a different cache format")
    reusable = {} if previous is None else {entry.file: entry for entry in previous.entries}
    runtime = _numeric_runtime_sha256()
    entries: list[CacheEntry] = []
    for decoded in progress(inputs, "[cache] MiniMax-H3 windows"):
        item = build_cache_item(decoded, format, policy)
        files = _source_files(decoded)
        identity = CacheIdentity(
            source_sha256=canonical_sha256(
                {
                    "files": dict(files),
                    "caption": decoded.record.caption,
                    "target_pixels": _tensor_sha256(decoded.frames),
                    "target_waveform": _tensor_sha256(decoded.waveform),
                    "controls": [_tensor_sha256(value) for value in decoded.control_frames],
                    "references": [
                        {
                            "kind": ref.kind,
                            "path": str(ref.path.resolve()),
                            "audio": None
                            if ref.audio is None
                            else {"path": str(ref.audio.path.resolve()), "embedded": ref.audio.embedded},
                        }
                        for ref in decoded.record.references
                    ],
                    "audio_source": None
                    if decoded.audio_source is None
                    else {"path": str(decoded.audio_source.path.resolve()), "embedded": decoded.audio_source.embedded},
                }
            ),
            encoder_sha256=encoder_sha256,
            transform_sha256=canonical_sha256(
                {
                    "schema": "minimax-h3/complete-student-cache/1",
                    "format": format.model_dump(mode="json"),
                    "item": item.model_dump(mode="json"),
                    "policy": policy.model_dump(mode="json"),
                    "canonical_item_key": _canonical_item_key(decoded, format),
                    "execution_sha256": encoders.execution_sha256,
                    "numeric_runtime": runtime,
                    "pixels": "Lanczos/half-up-center-crop;uint8FP32/127.5-1;floatingFP32*2-1;alpha-drop",
                    "posterior": "target-SHA256-CPU;condition42-FP16-FP32-before-normalization;audio-mean",
                    "reference": "native24Hz/17n+5/32-grid-Lanczos;image-target-area;single-cap360;text2Hz-shared-pixels",
                }
            ),
            layout=format.layout,
        )
        entry = reusable.get(identity.digest() + ".safetensors")
        if entry is None:
            path = destination / f"{identity.digest()}.safetensors"
            if path.is_file():
                entry = CacheEntry(file=path.name, sha256=file_sha256(path))
        if entry is not None:
            stored = read_minimax_h3_record(destination, entry, format=format, device=torch.device("cpu"))
            if stored.item != item:
                raise ValueError("MiniMaxH3 reused cache item differs from its requested identity")
        else:
            record = encode_cache_record(decoded, encoders, format=format, policy=policy)
            if files != _source_files(decoded):
                raise ValueError("MiniMaxH3 source media changed during cache encoding")
            if validate_encoder_artifacts is not None:
                validate_encoder_artifacts()
            entry = save_minimax_h3_record(destination, identity, record, format=format)
        entries.append(entry)
        if repa is not None:
            repa.publish(entry, decoded.frames)
    manifest = MiniMaxH3CacheManifest(**format.model_dump(), entries=tuple(entries))
    if validate_encoder_artifacts is not None:
        validate_encoder_artifacts()
    save_minimax_h3_manifest(manifest, destination)
    return manifest
