"""Resolve H3 generation media and encode shared native conditions without a dummy target."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from unituner.config.audio import AudioDecodeConfig
from unituner.models.minimax_h3.conditioning import encode_audio, encode_condition_video
from unituner.models.minimax_h3.config.inputs import MiniMaxH3InputsConfig, validate_minimax_h3_inputs
from unituner.models.minimax_h3.config.sampling import MiniMaxH3SamplingConfig
from unituner.models.minimax_h3.data.encoding import encode_reference_conditions
from unituner.models.minimax_h3.data.media.inputs import prepare_pixels, read_condition_image
from unituner.models.minimax_h3.data.sources import MiniMaxH3Reference, resolve_references
from unituner.models.minimax_h3.inference.sampling import MiniMaxH3VideoInputs
from unituner.models.minimax_h3.layout import (
    FRAME_RESCALE,
    MiniMaxH3Task,
    MiniMaxH3TimeOverrides,
    MiniMaxH3VideoGeometry,
    audio_latent_frames,
    build_layout,
    one_frame_condition_roles,
    video_latent_frames,
)
from unituner.models.minimax_h3.text.encoder import MiniMaxH3TextEncoder
from unituner.models.minimax_h3.text.presentation import MiniMaxH3TextVisual, build_presentation
from unituner.models.minimax_h3.vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE


@dataclass(frozen=True, slots=True)
class MiniMaxH3ResolvedInputs:
    """Own resolved ordered reference/anchor paths and explicit native rotary coordinates."""

    task: MiniMaxH3Task
    prompt: str
    anchors: tuple[tuple[str, Path], ...]
    references: tuple[MiniMaxH3Reference, ...]
    time_overrides: MiniMaxH3TimeOverrides | None


def resolve_minimax_h3_inputs(
    inputs: MiniMaxH3InputsConfig,
    sampling: MiniMaxH3SamplingConfig,
    *,
    task: str,
    directory: Path,
) -> MiniMaxH3ResolvedInputs:
    """Preflight actual local media paths/streams and task policy before model loading.

    Relative paths resolve against explicit directory; no target, fake cache
    record or target posterior is constructed. Keep unsorted/repeated controls
    and reference order. Probing validates clocks,2-15second reference video and
    actual audio-bearing reference limits, including omitted embedded audio.
    """
    validate_minimax_h3_inputs(inputs, sampling, task=task)
    if task not in ("t2va", "fl2va", "ref2va"):
        raise ValueError("MiniMaxH3 generation task must be t2va/fl2va/ref2va")
    references = resolve_references(inputs.references, directory, task, one_frame=sampling.frames == 1)
    entries = []
    if sampling.frames == 1:
        roles = one_frame_condition_roles(len(inputs.controls))
        entries = list(zip(roles, (control.path for control in inputs.controls), strict=True))
    else:
        entries = [
            (role, path)
            for role, path in [("first", inputs.first_frame), ("last", inputs.last_frame)]
            if path is not None
        ]
    anchors = []
    for role, value in entries:
        path = Path(value)
        path = (directory / path).resolve() if not path.is_absolute() else path.resolve()
        if not value.strip() or not path.is_file():
            raise ValueError(f"MiniMaxH3 condition {role} requires an existing image file: {path}")
        anchors.append((role, path))
    times = None
    if sampling.frames == 1:
        times = MiniMaxH3TimeOverrides(
            tuple(FRAME_RESCALE * control.index for control in inputs.controls),
            FRAME_RESCALE * (0 if inputs.target_index is None else inputs.target_index),
        )
    return MiniMaxH3ResolvedInputs(task, inputs.prompt, tuple(anchors), references, times)


@torch.no_grad()
def build_minimax_h3_inputs(
    resolved: MiniMaxH3ResolvedInputs,
    sampling: MiniMaxH3SamplingConfig,
    video: MiniMaxH3VideoVAE,
    text: MiniMaxH3TextEncoder,
    *,
    audio: MiniMaxH3AudioVAE | None = None,
) -> MiniMaxH3VideoInputs:
    """Encode prepared media/text with frozen FP32 owners and native condition sampling.

    Borrow VAEs on their caller-selected CPU/CUDA devices without moving or
    changing modes/weights/RNG. Encode visual posterior samples with local seed42,
    round FP16→FP32 before normalization, and deterministic audio means. Video
    references share decoding with text2Hz presentation. Return owned CPU batched
    latent/text/tag tensors and native packed geometry. Audio decode uses32kHz
    stereo and the shared800sample tail/6400sample repair policy.
    """
    vaes = (video,) if audio is None else (video, audio)
    if (
        any(module.training for vae in vaes for module in vae.modules())
        or any(
            parameter.requires_grad or parameter.dtype != torch.float32 or parameter.device.type not in ("cpu", "cuda")
            for vae in vaes
            for parameter in vae.parameters()
        )
        or any(
            value.device != next(vae.parameters()).device or value.dtype != torch.float32
            for vae in vaes
            for value in vae.buffers()
        )
    ):
        raise ValueError("MiniMaxH3 condition vaes must be frozen eval FP32 on CPU/CUDA")
    if video.vae_ratio != 16 or video.vae_ratio_t != 4 or video.clip_length != 17 or video.token_drop != 3:
        raise ValueError("MiniMaxH3 condition vae requires native16x/4x with17frame clips/drop3")
    video_device = next(video.parameters()).device

    def visual(pixels: Tensor) -> Tensor:
        with torch.autocast(video_device.type, enabled=False):
            return encode_condition_video(video, pixels.to(video_device))

    def waveform(samples: Tensor) -> Tensor:
        if audio is None:
            raise ValueError("MiniMaxH3 audio-bearing references require an audio vae")
        with torch.autocast(next(audio.parameters()).device.type, enabled=False):
            return encode_audio(audio, samples)

    single = sampling.frames == 1
    anchors = {
        role: MiniMaxH3TextVisual(read_condition_image(path, resolution=(sampling.width, sampling.height)))
        for role, path in resolved.anchors
    }
    visuals = tuple(visual(prepare_pixels(item.frames)) for item in anchors.values())
    reference_conditions = encode_reference_conditions(
        resolved.references,
        frame_cap=360 if single else sampling.frames * 24 // sampling.fps,
        resolution=(sampling.width, sampling.height),
        standalone_audio_samples=1600
        if single
        else audio_latent_frames(sampling.frames, output_fps=sampling.fps) * 800,
        condition_video=visual,
        audio=waveform,
        audio_decode=AudioDecodeConfig(sample_rate=32000, channels=2),
    )
    presentation = build_presentation(
        resolved.prompt, resolved.task, anchors=anchors, references=reference_conditions.text_references
    )
    features = text.encode(presentation)
    layout = build_layout(
        task=resolved.task,
        text_length=features.hidden_states.shape[0],
        target_video=(
            1 if single else video_latent_frames(sampling.frames),
            sampling.height // 16,
            sampling.width // 16,
        ),
        target_audio_frames=2 if single else audio_latent_frames(sampling.frames, output_fps=sampling.fps),
        visual_conditions=tuple(MiniMaxH3VideoGeometry(*value.shape[2:]) for value in visuals),
        references=reference_conditions.geometries,
        one_frame=single,
        condition_roles=tuple(anchors) if resolved.task == "fl2va" and not single else None,
        time_overrides=resolved.time_overrides,
        output_fps=sampling.fps,
        temporal_fine_bands=sampling.temporal_fine_bands,
    )
    return MiniMaxH3VideoInputs(
        features.hidden_states.detach()
        .to(
            device="cpu",
            dtype=torch.float32 if features.hidden_states.dtype == torch.float16 else features.hidden_states.dtype,
            copy=True,
        )
        .unsqueeze(0),
        features.token_tags.detach().to("cpu", copy=True).unsqueeze(0),
        layout,
        tuple(value.detach().to("cpu", copy=True) for value in visuals)
        + tuple(value.detach().to("cpu", copy=True).unsqueeze(0) for value in reference_conditions.visual_conditions),
        tuple(value.detach().to("cpu", copy=True).unsqueeze(0) for value in reference_conditions.audio_conditions),
    )
