"""Draw H3 training noise in native video, uniform, audio and condition order."""

import torch
from torch import Tensor

from unitunner.models.minimax_h3.config.training import MiniMaxH3FlowConfig
from unitunner.models.minimax_h3.data.cache.storage import MiniMaxH3CacheFormat, MiniMaxH3CacheRecord
from unitunner.training.integrations.minimax_h3.binding import MiniMaxH3FlowRequest, build_cached_request


def _condition_noise(values: tuple[Tensor, ...], clean: float) -> tuple[Tensor, ...]:
    """Draw independent FP32 batched condition noise only for a non-clean group."""
    if clean == 1.0:
        return ()
    return tuple(torch.randn((1, *value.shape), dtype=torch.float32, device=value.device)[0] for value in values)


def sample_minimax_h3_request(
    record: MiniMaxH3CacheRecord, *, format: MiniMaxH3CacheFormat, config: MiniMaxH3FlowConfig
) -> MiniMaxH3FlowRequest:
    """Consume the native per-device global RNG stream for one verified FP32 record.

    Draw batched video noise, one FP32 uniform, batched audio noise, then visual
    and audio condition noise in their declared order. Fully clean groups draw
    nothing; absent audio still consumes target audio noise and executes the
    audio branch. The caller restores global RNG before the first resumed draw.
    """
    video_noise = torch.randn_like(record.video.unsqueeze(0))[0]
    uniform = torch.rand((1,), device=record.video.device)
    audio_noise = torch.randn_like(record.audio.unsqueeze(0))[0]
    visual = _condition_noise(record.visual_conditions, config.visual_condition_clean)
    audio = _condition_noise(record.audio_conditions, config.audio_condition_clean)
    return build_cached_request(
        record,
        format=format,
        video_noise=video_noise,
        audio_noise=audio_noise,
        uniform=uniform,
        visual_condition_noise=visual,
        audio_condition_noise=audio,
    )
