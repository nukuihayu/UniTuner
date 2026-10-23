# Copyright 2025-2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Native joint Euler sampling follows Diffusers PR #14355 via pinned Musubi.
"""Integrate native H3 video/audio velocities with independent shifted FP64 schedules."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from unitunner.execution.attention import inference_attention
from unitunner.models.minimax_h3.config.sampling import MiniMaxH3SamplingConfig
from unitunner.models.minimax_h3.data.media.output import (
    MiniMaxH3DecodedAV,
    MiniMaxH3DecodedImage,
    synchronize_decoded_av,
)
from unitunner.models.minimax_h3.layout import (
    MiniMaxH3PackedLayout,
    MiniMaxH3VideoGeometry,
    audio_latent_frames,
    video_latent_frames,
)
from unitunner.models.minimax_h3.packing import MiniMaxH3Targets
from unitunner.models.minimax_h3.transformer import MiniMaxH3Transformer
from unitunner.models.minimax_h3.vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE

_CPU = torch.device("cpu")


@dataclass(frozen=True, slots=True)
class MiniMaxH3Schedule:
    """Own FP64 base/video/audio noise fractions from pure noise1 to clean0."""

    base: Tensor
    video: Tensor
    audio: Tensor


def build_schedule(
    steps: int, *, video_shift: float = 12.0, audio_shift: float = 3.0, device: torch.device = _CPU
) -> MiniMaxH3Schedule:
    """Build native FP641→0 fractions without a training-time1000 roundtrip.

    Keep linspace, shift arithmetic and adjacent deltas in FP64. The sampler
    casts model times and each modality's step delta independently to FP32.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError("MiniMaxH3 sampling steps must be a positive integer")
    if any(not math.isfinite(shift) or not 0.01 <= shift <= 100 for shift in (video_shift, audio_shift)):
        raise ValueError("MiniMaxH3 sampling shifts must be finite in[0.01,100]")
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64, device=device)
    video = video_shift * base / (1.0 + (video_shift - 1.0) * base)
    audio = audio_shift * base / (1.0 + (audio_shift - 1.0) * base)
    return MiniMaxH3Schedule(base, video, audio)


def _validate_initial(
    model: MiniMaxH3Transformer,
    video: Tensor,
    audio: Tensor,
    text: Tensor,
    tags: Tensor,
    layout: MiniMaxH3PackedLayout,
    conditions: tuple[Tensor, ...],
    allow_cpu_frozen: bool,
) -> None:
    """Validate finite native inputs and resident FP32 model ownership once before sampling."""
    expected_video = (1, 24, layout.target_video.frames, layout.target_video.height, layout.target_video.width)
    if tuple(video.shape) != expected_video or tuple(audio.shape) != (1, 32, 2, layout.target_audio_frames):
        raise ValueError("MiniMaxH3 sampling requires native B=1 target geometry matching the layout")
    if text.shape != (1, layout.text_length, model.config.text_dim) or tags.shape != (1, layout.text_length):
        raise ValueError("MiniMaxH3 sampling text/tags differ from layout and model text width")
    if text.dtype not in (torch.float32, torch.bfloat16) or not torch.isfinite(text).all():
        raise ValueError("MiniMaxH3 sampling text must be finite FP32 or native BF16 rows")
    if tags.dtype != torch.int64:
        raise ValueError("MiniMaxH3 sampling modality tags must be int64")
    if any(
        value.dtype != torch.float32 or value.device != video.device or not torch.isfinite(value).all()
        for value in (video, audio, *conditions)
    ):
        raise ValueError("MiniMaxH3 sampling targets/conditions must be finite FP32 on one device")
    if any(
        parameter.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or (
            parameter.device != video.device
            and not (allow_cpu_frozen and parameter.device.type == "cpu" and not parameter.requires_grad)
        )
        for parameter in model.parameters()
    ):
        raise ValueError("MiniMaxH3 sampling requires a resident floating model on the target device")


@torch.no_grad()
def sample_latents(
    model: MiniMaxH3Transformer,
    video_noise: Tensor,
    audio_noise: Tensor,
    text_hidden_states: Tensor,
    text_token_tags: Tensor,
    layout: MiniMaxH3PackedLayout,
    *,
    steps: int,
    video_shift: float = 12.0,
    audio_shift: float = 3.0,
    visual_conditions: tuple[Tensor, ...] = (),
    audio_conditions: tuple[Tensor, ...] = (),
    visual_condition_clean: float = 0.999,
    audio_condition_clean: float = 1.0,
    step_callback: Callable[[int, int], None] | None = None,
    x0_callback: Callable[[int, Tensor, Tensor], None] | None = None,
    allow_cpu_frozen: bool = False,
) -> MiniMaxH3Targets:
    """Integrate explicit FP32 B24FHW/B32ST noise with native dataward Euler updates.

    Model clean times are1-sigma. Preserve video/audio independent FP64 schedule
    deltas before casting to target dtype: x_next=x+delta*(x0-eps). Conditions
    are already augmented and remain fixed across steps. Borrow model mode,
    RNG, parameters and inputs without changing them. Disable ambient autocast.
    Optional x0 snapshots use x+sigma*velocity before each update; callbacks
    receive independent estimates, not live sampling storage.
    """
    _validate_initial(
        model,
        video_noise,
        audio_noise,
        text_hidden_states,
        text_token_tags,
        layout,
        (*visual_conditions, *audio_conditions),
        allow_cpu_frozen,
    )
    if any(
        not math.isfinite(clean) or not 0 <= clean <= 1 for clean in (visual_condition_clean, audio_condition_clean)
    ):
        raise ValueError("MiniMaxH3 sampling condition clean coefficients must be finite in[0,1]")
    schedule = build_schedule(steps, video_shift=video_shift, audio_shift=audio_shift, device=video_noise.device)
    video, audio = video_noise, audio_noise
    with torch.autocast(
        video.device.type,
        dtype=model.dtype if model.dtype != torch.float32 else torch.bfloat16,
        enabled=model.dtype != torch.float32,
        cache_enabled=False,
    ):
        for index in range(steps):
            sigma_video = schedule.video[index].to(dtype=torch.float32)
            sigma_audio = schedule.audio[index].to(dtype=torch.float32)
            prediction = model(
                video_latents=video,
                audio_latents=audio,
                text_hidden_states=text_hidden_states,
                text_token_tags=text_token_tags,
                layout=layout,
                model_t_video=1.0 - sigma_video,
                model_t_audio=1.0 - sigma_audio,
                visual_condition_latents=visual_conditions,
                audio_condition_latents=audio_conditions,
                visual_condition_clean=visual_condition_clean,
                audio_condition_clean=audio_condition_clean,
            )
            if prediction.video.shape != video.shape or prediction.audio.shape != audio.shape:
                raise ValueError("MiniMaxH3 sampling prediction differs from target geometry")
            if x0_callback is not None:
                x0_callback(
                    index,
                    video + sigma_video.to(video) * prediction.video,
                    audio + sigma_audio.to(audio) * prediction.audio,
                )
            video_delta = (schedule.video[index] - schedule.video[index + 1]).to(video)
            audio_delta = (schedule.audio[index] - schedule.audio[index + 1]).to(audio)
            video = video + video_delta * prediction.video
            audio = audio + audio_delta * prediction.audio
            if step_callback is not None:
                step_callback(index + 1, steps)
    return MiniMaxH3Targets(video, audio)


def _augment_conditions(
    values: tuple[Tensor, ...], *, generator: torch.Generator, clean: float, device: torch.device
) -> tuple[Tensor, ...]:
    """Consume the seed-local CPU noise stream in native condition order."""
    moved = tuple(value.to(device) for value in values)
    if clean == 1.0:
        return moved
    return tuple(
        clean * value
        + (1.0 - clean)
        * torch.randn(tuple(value.shape), generator=generator, dtype=torch.float32, device="cpu").to(value)
        for value in moved
    )


@torch.no_grad()
def sample_seeded_latents(
    model: MiniMaxH3Transformer,
    text_hidden_states: Tensor,
    text_token_tags: Tensor,
    layout: MiniMaxH3PackedLayout,
    *,
    seed: int,
    steps: int,
    video_shift: float = 12.0,
    audio_shift: float = 3.0,
    visual_conditions: tuple[Tensor, ...] = (),
    audio_conditions: tuple[Tensor, ...] = (),
    visual_condition_clean: float = 0.999,
    audio_condition_clean: float = 1.0,
    step_callback: Callable[[int, int], None] | None = None,
    x0_callback: Callable[[int, Tensor, Tensor], None] | None = None,
    allow_cpu_frozen: bool = False,
) -> MiniMaxH3Targets:
    """Draw one CPU seed stream and return owned CPU FP32 joint sampling results.

    Preserve original video→audio→visual conditions→audio conditions RNG order.
    Original generation rounds text to BF16 before model execution, even for an
    FP32 model; preserve that distinct inference boundary. Global RNG is untouched.
    """
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    video = torch.randn(
        (1, 24, layout.target_video.frames, layout.target_video.height, layout.target_video.width),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device)
    audio = torch.randn(
        (1, 32, 2, layout.target_audio_frames), generator=generator, dtype=torch.float32, device="cpu"
    ).to(device)
    visuals = _augment_conditions(visual_conditions, generator=generator, clean=visual_condition_clean, device=device)
    audios = _augment_conditions(audio_conditions, generator=generator, clean=audio_condition_clean, device=device)
    result = sample_latents(
        model,
        video,
        audio,
        text_hidden_states.to(device=device, dtype=torch.bfloat16),
        text_token_tags.to(device),
        layout,
        steps=steps,
        video_shift=video_shift,
        audio_shift=audio_shift,
        visual_conditions=visuals,
        audio_conditions=audios,
        visual_condition_clean=visual_condition_clean,
        audio_condition_clean=audio_condition_clean,
        step_callback=step_callback,
        x0_callback=x0_callback,
        allow_cpu_frozen=allow_cpu_frozen,
    )
    return MiniMaxH3Targets(result.video.detach().to("cpu", copy=True), result.audio.detach().to("cpu", copy=True))


@dataclass(frozen=True, slots=True)
class MiniMaxH3VideoInputs:
    """Borrow batched native text/tags and condition latents under one immutable packed layout."""

    text: Tensor
    tags: Tensor
    layout: MiniMaxH3PackedLayout
    visual_conditions: tuple[Tensor, ...] = ()
    audio_conditions: tuple[Tensor, ...] = ()


def _validate_vae(vae: MiniMaxH3VideoVAE | MiniMaxH3AudioVAE) -> None:
    """Require borrowed frozen eval CPU FP32 owners before sampling or residency changes."""
    if (
        any(module.training for module in vae.modules())
        or any(
            parameter.requires_grad or parameter.device.type != "cpu" or parameter.dtype != torch.float32
            for parameter in vae.parameters()
        )
        or any(value.device.type != "cpu" or value.dtype != torch.float32 for value in vae.buffers())
    ):
        raise ValueError("MiniMaxH3 render vaes must be frozen eval CPU FP32")
    if vae.decoder is None:
        raise ValueError("MiniMaxH3 render vaes require their decoders")


@torch.no_grad()
def render_minimax_h3(
    model: MiniMaxH3Transformer,
    video_vae: MiniMaxH3VideoVAE,
    sampling: MiniMaxH3SamplingConfig,
    inputs: MiniMaxH3VideoInputs,
    *,
    audio_vae: MiniMaxH3AudioVAE | None = None,
    allow_cpu_frozen: bool = False,
) -> MiniMaxH3DecodedImage | MiniMaxH3DecodedAV:
    """Sample jointly and return owned CPU RGB or synchronized RGB/stereo media.

    Borrow a resident FP32 transformer and frozen eval CPU FP32 VAEs. Decode
    video then audio, moving one VAE at a time to the transformer device and
    restoring it to CPU even after failure. Single-frame output samples both
    branches but never invokes the audio decoder. Preserve global RNG, model
    modes, weights and supplied conditions; disable ambient decode autocast.
    This explicit FP32 path differs from the official BF16 video decoder default.
    """
    single = sampling.frames == 1
    layout = inputs.layout
    expected_video = MiniMaxH3VideoGeometry(
        1 if single else video_latent_frames(sampling.frames), sampling.height // 16, sampling.width // 16
    )
    expected_audio = 2 if single else audio_latent_frames(sampling.frames, output_fps=sampling.fps)
    if (
        layout.target_video != expected_video
        or layout.target_audio_frames != expected_audio
        or layout.output_fps != sampling.fps
        or layout.temporal_fine_bands != sampling.temporal_fine_bands
    ):
        raise ValueError("MiniMaxH3 render sampling geometry/timeline differs from prepared layout")
    if sampling.temporal_fine_bands >= model.config.rope_inv_freq_len:
        raise ValueError("MiniMaxH3 temporal_fine_bands must leave at least one stretched rotary band")
    _validate_vae(video_vae)
    if (
        video_vae.vae_ratio != 16
        or video_vae.vae_ratio_t != 4
        or video_vae.clip_length != 17
        or video_vae.token_drop != 3
    ):
        raise ValueError("MiniMaxH3 render requires native16x spatial/4x temporal vae with17-frame clips/drop3")
    if not single:
        if audio_vae is None:
            raise ValueError("MiniMaxH3 video render requires an audio vae")
        _validate_vae(audio_vae)
    with inference_attention(sampling.attention_backend):
        latents = sample_seeded_latents(
            model,
            inputs.text,
            inputs.tags,
            layout,
            seed=sampling.seed,
            steps=sampling.steps,
            video_shift=sampling.video_shift,
            audio_shift=sampling.audio_shift,
            visual_conditions=inputs.visual_conditions,
            audio_conditions=inputs.audio_conditions,
            visual_condition_clean=sampling.visual_condition_clean,
            audio_condition_clean=sampling.audio_condition_clean,
            allow_cpu_frozen=allow_cpu_frozen,
        )
    device = next(model.parameters()).device
    try:
        video_vae.to(device)
        with torch.autocast(device.type, enabled=False):
            pixels = video_vae.decode(latents.video.to(device)).detach().cpu()
    finally:
        video_vae.to("cpu")
    if (
        pixels.ndim != 5
        or pixels.shape[:2] != (1, 3)
        or pixels.shape[2] < 1
        or pixels.shape[3:] != (sampling.height, sampling.width)
        or not torch.isfinite(pixels).all()
    ):
        raise ValueError("MiniMaxH3 decoded video differs from requested RGB geometry or is nonfinite")
    if single:
        image = ((pixels[0, :, 0].float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
        return MiniMaxH3DecodedImage(image.permute(1, 2, 0).contiguous())
    assert audio_vae is not None
    try:
        audio_vae.to(device)
        with torch.autocast(device.type, enabled=False):
            waveform = audio_vae.decode(latents.audio.to(device)).detach().cpu()
    finally:
        audio_vae.to("cpu")
    return synchronize_decoded_av(pixels, waveform, frame_count=sampling.frames, fps=sampling.fps)
