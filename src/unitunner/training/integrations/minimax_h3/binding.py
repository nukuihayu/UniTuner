# Copyright 2025-2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Native flow formulas follow the pinned Musubi and old MixTuner H3 callers.
"""Bind native H3 audio/video flow math to the shared training engine."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unitunner.execution.regions import ModelRegion, ModelRegions
from unitunner.models.minimax_h3.config.training import MiniMaxH3FlowConfig
from unitunner.models.minimax_h3.data.cache.storage import (
    MiniMaxH3CacheFormat,
    MiniMaxH3CacheRecord,
    build_record_layout,
)
from unitunner.models.minimax_h3.layout import MiniMaxH3PackedLayout
from unitunner.models.minimax_h3.packing import MiniMaxH3Targets
from unitunner.models.minimax_h3.projections import adapter_projections
from unitunner.models.minimax_h3.transformer import MiniMaxH3Transformer
from unitunner.training.contracts import StepResult


@dataclass(frozen=True, slots=True)
class MiniMaxH3FlowRequest:
    """Supply FP32 BCFHW video, BCST stereo audio and explicit independent noise.

    R1 has B=1. Uniform draws are B FP32 values in [0,1). Conditions follow the
    layout's modality order; their noises are mandatory only for clean<1. No RNG
    is drawn here. Audio presence controls supervision, not model execution.
    """

    video: Tensor
    video_noise: Tensor
    audio: Tensor
    audio_noise: Tensor
    uniform: Tensor
    text_hidden_states: Tensor
    text_token_tags: Tensor
    layout: MiniMaxH3PackedLayout
    visual_conditions: tuple[Tensor, ...] = ()
    audio_conditions: tuple[Tensor, ...] = ()
    visual_condition_noise: tuple[Tensor, ...] = ()
    audio_condition_noise: tuple[Tensor, ...] = ()
    audio_present: bool = True


@dataclass(frozen=True, slots=True)
class MiniMaxH3NoisyInputs:
    """Separate pre-roundtrip noise fractions, trainer units and native clean time.

    Video noising uses the initial base. Audio noising and both model times use
    the recovered base, preserving the original FP32 (base*1000+1-1)/1000 order.
    Native clean time is 0=noise, 1=clean. Trainer time is base*1000+1.
    """

    base_sigma: Tensor
    trainer_timesteps: Tensor
    recovered_base_sigma: Tensor
    video_sigma: Tensor
    audio_sigma: Tensor
    model_t_video: Tensor
    model_t_audio: Tensor
    video: Tensor
    audio: Tensor
    visual_conditions: tuple[Tensor, ...]
    audio_conditions: tuple[Tensor, ...]


@dataclass(frozen=True, slots=True)
class MiniMaxH3FlowPrediction:
    """Return native clean-minus-noise velocities and differentiable noisy inputs."""

    prediction: MiniMaxH3Targets
    target: MiniMaxH3Targets
    inputs: MiniMaxH3NoisyInputs


def build_cached_request(
    record: MiniMaxH3CacheRecord,
    *,
    format: MiniMaxH3CacheFormat,
    video_noise: Tensor,
    audio_noise: Tensor,
    uniform: Tensor,
    visual_condition_noise: tuple[Tensor, ...] = (),
    audio_condition_noise: tuple[Tensor, ...] = (),
) -> MiniMaxH3FlowRequest:
    """Borrow one verified cache record and explicit unbatched noise for native flow.

    Add only the leading B=1 axis. Preserve FP32 values, CPU modality tags,
    reference order, native single-frame coordinates and absent-audio VAE
    values. No RNG, normalization, repacking, mask construction or dtype cast
    occurs here. Condition noise follows the separate visual/audio inventories;
    the flow policy validates whether each inventory is required.
    """
    layout = build_record_layout(record, format=format)
    _validate_noise(record.video, video_noise, "cached video")
    _validate_noise(record.audio, audio_noise, "cached audio")
    if visual_condition_noise and len(visual_condition_noise) != len(record.visual_conditions):
        raise ValueError("H3 cached visual condition noise inventory differs")
    if audio_condition_noise and len(audio_condition_noise) != len(record.audio_conditions):
        raise ValueError("H3 cached audio condition noise inventory differs")
    if visual_condition_noise:
        for clean, noise in zip(record.visual_conditions, visual_condition_noise, strict=True):
            _validate_noise(clean, noise, "cached visual condition")
    if audio_condition_noise:
        for clean, noise in zip(record.audio_conditions, audio_condition_noise, strict=True):
            _validate_noise(clean, noise, "cached audio condition")
    return MiniMaxH3FlowRequest(
        video=record.video.unsqueeze(0),
        video_noise=video_noise.unsqueeze(0),
        audio=record.audio.unsqueeze(0),
        audio_noise=audio_noise.unsqueeze(0),
        uniform=uniform,
        text_hidden_states=record.text.unsqueeze(0),
        text_token_tags=record.token_tags.unsqueeze(0),
        layout=layout,
        visual_conditions=tuple(value.unsqueeze(0) for value in record.visual_conditions),
        audio_conditions=tuple(value.unsqueeze(0) for value in record.audio_conditions),
        visual_condition_noise=tuple(value.unsqueeze(0) for value in visual_condition_noise),
        audio_condition_noise=tuple(value.unsqueeze(0) for value in audio_condition_noise),
        audio_present=record.item.audio_present,
    )


def _validate_noise(clean: Tensor, noise: Tensor, label: str) -> None:
    if (
        clean.dtype != torch.float32
        or noise.dtype != clean.dtype
        or noise.shape != clean.shape
        or noise.device != clean.device
    ):
        raise ValueError(f"H3 {label} requires matching FP32 clean/noise tensors on one device")


def _shift_sigma(base: Tensor, shift: float) -> Tensor:
    """Apply the native rational shift without regrouping FP32 operations."""
    return shift * base / (1.0 + (shift - 1.0) * base)


def _augment_conditions(conditions: tuple[Tensor, ...], noises: tuple[Tensor, ...], clean: float) -> tuple[Tensor, ...]:
    """Blend supplied noises in modality order; fully clean groups consume none."""
    if clean == 1.0:
        if noises:
            raise ValueError("H3 fully clean conditions must not supply unused noise")
        return conditions
    if len(conditions) != len(noises):
        raise ValueError("H3 condition noise inventory must match its modality group")
    for condition, noise in zip(conditions, noises, strict=True):
        _validate_noise(condition, noise, "condition")
    return tuple(clean * condition + (1.0 - clean) * noise for condition, noise in zip(conditions, noises, strict=True))


def build_noisy_inputs(request: MiniMaxH3FlowRequest, config: MiniMaxH3FlowConfig) -> MiniMaxH3NoisyInputs:
    """Build deterministic native FP32 noising with the original trainer roundtrip.

    The caller supplies all RNG results. Validate one sample here; the model
    validates packed geometry and text. Inputs and their gradient paths remain
    intact. Low precision has not been accepted at this boundary.
    """
    _validate_noise(request.video, request.video_noise, "video")
    _validate_noise(request.audio, request.audio_noise, "audio")
    if request.video.ndim != 5 or request.video.shape[0] != 1 or request.audio.ndim != 4 or request.audio.shape[0] != 1:
        raise ValueError("H3 R1 flow requires B=1 video BCFHW and audio BCST")
    if request.audio.device != request.video.device or type(request.audio_present) is not bool:
        raise ValueError("H3 modalities must share a device and audio_present must be boolean")
    u = request.uniform
    if (
        u.shape != (1,)
        or u.dtype != torch.float32
        or u.device != request.video.device
        or not torch.all(torch.isfinite(u) & (u >= 0) & (u < 1))
    ):
        raise ValueError("H3 uniform requires one finite FP32 draw in [0,1) on the latent device")
    base = config.sigma_min + (config.sigma_max - config.sigma_min) * u
    if config.focus_probability > 0.0:
        focused = config.focus_min + (config.focus_max - config.focus_min) * (u / config.focus_probability)
        passthrough = config.sigma_min + (config.sigma_max - config.sigma_min) * (
            (u - config.focus_probability) / max(1.0 - config.focus_probability, 1e-8)
        )
        base = torch.where(u < config.focus_probability, focused, passthrough)
    video_sigma = _shift_sigma(base, config.video_shift)
    video_fraction = video_sigma.view(-1, 1, 1, 1, 1)
    video = ((1.0 - video_fraction) * request.video.float() + video_fraction * request.video_noise.float()).to(
        request.video.dtype
    )
    trainer_timesteps = base * 1000.0 + 1.0
    recovered = (trainer_timesteps - 1.0) / 1000.0
    recovered_video_sigma = _shift_sigma(recovered, config.video_shift)
    audio_sigma = _shift_sigma(recovered, config.audio_shift)
    audio_fraction = audio_sigma.view(-1, 1, 1, 1)
    audio = (1.0 - audio_fraction) * request.audio + audio_fraction * request.audio_noise
    return MiniMaxH3NoisyInputs(
        base,
        trainer_timesteps,
        recovered,
        video_sigma,
        audio_sigma,
        1.0 - recovered_video_sigma,
        1.0 - audio_sigma,
        video,
        audio,
        _augment_conditions(request.visual_conditions, request.visual_condition_noise, config.visual_condition_clean),
        _augment_conditions(request.audio_conditions, request.audio_condition_noise, config.audio_condition_clean),
    )


class MiniMaxH3Binding:
    """Borrow the joint AV model; expose deterministic baseline flow and regions.

    Sampling, cache ownership and parameter installation belong to shared
    application owners. Guidance and teacher matching require separate bindings.
    """

    def __init__(self, model: MiniMaxH3Transformer, config: MiniMaxH3FlowConfig) -> None:
        self.model = model
        self.config = config
        blocks = tuple(ModelRegion(block, index, "joint") for index, block in enumerate(model.blocks))
        self.regions = ModelRegions(blocks)

    def adapter_targets(self) -> dict[str, nn.Linear]:
        """Borrow the accepted joint attention and FFN projection inventory."""
        return adapter_projections(self.model)

    def predict(self, request: MiniMaxH3FlowRequest) -> MiniMaxH3FlowPrediction:
        """Predict native clean-minus-noise velocities without altering RNG or inputs."""
        inputs = build_noisy_inputs(request, self.config)
        prediction = self.model(
            video_latents=inputs.video,
            audio_latents=inputs.audio,
            text_hidden_states=request.text_hidden_states,
            text_token_tags=request.text_token_tags,
            layout=request.layout,
            model_t_video=inputs.model_t_video,
            model_t_audio=inputs.model_t_audio,
            visual_condition_latents=inputs.visual_conditions,
            audio_condition_latents=inputs.audio_conditions,
            visual_condition_clean=self.config.visual_condition_clean,
            audio_condition_clean=self.config.audio_condition_clean,
        )
        target = MiniMaxH3Targets(request.video - request.video_noise, request.audio - request.audio_noise)
        return MiniMaxH3FlowPrediction(prediction, target, inputs)

    def loss(self, result: MiniMaxH3FlowPrediction, *, audio_present: bool) -> Tensor:
        """Sum independent FP32 modality means; skip unsupervised audio arithmetic."""
        if type(audio_present) is not bool:
            raise ValueError("H3 audio_present must be boolean")
        video_loss = F.mse_loss(result.prediction.video.float(), result.target.video.float(), reduction="mean")
        weight = self.config.audio_loss_weight if audio_present else 0.0
        audio_loss = (
            video_loss.detach().new_zeros(())
            if weight == 0.0
            else F.mse_loss(result.prediction.audio.float(), result.target.audio.float(), reduction="mean")
        )
        return video_loss + weight * audio_loss

    def step(self, request: MiniMaxH3FlowRequest) -> StepResult:
        """Return the differentiable single-sample loss for shared accumulation."""
        return StepResult(self.loss(self.predict(request), audio_present=request.audio_present), 1)
