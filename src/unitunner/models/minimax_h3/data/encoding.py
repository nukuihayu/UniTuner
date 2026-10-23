"""Encode ordered H3 references once for shared cache and generation consumers."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from unitunner.config.audio import AudioDecodeConfig
from unitunner.data.media.audio import decode_audio
from unitunner.data.media.audio_timeline import slice_audio_window
from unitunner.models.minimax_h3.data.media.inputs import prepare_pixels, read_reference_image, read_reference_video
from unitunner.models.minimax_h3.data.sources import MiniMaxH3Reference
from unitunner.models.minimax_h3.layout import MiniMaxH3ReferenceGeometry, MiniMaxH3VideoGeometry, audio_latent_frames
from unitunner.models.minimax_h3.text.presentation import MiniMaxH3TextReference, MiniMaxH3TextVisual


@dataclass(frozen=True, slots=True)
class MiniMaxH3ReferenceEncodings:
    """Own ordered unbatched VAE results and borrow decoded pixels for text presentation."""

    visual_conditions: tuple[Tensor, ...]
    audio_conditions: tuple[Tensor, ...]
    geometries: tuple[MiniMaxH3ReferenceGeometry, ...]
    text_references: tuple[MiniMaxH3TextReference, ...]


@torch.no_grad()
def encode_reference_conditions(
    references: Sequence[MiniMaxH3Reference],
    *,
    frame_cap: int,
    resolution: tuple[int, int],
    standalone_audio_samples: int,
    condition_video: Callable[[Tensor], Tensor],
    audio: Callable[[Tensor], Tensor],
    audio_decode: AudioDecodeConfig,
    audio_pad_tolerance: int = 800,
    audio_max_repair_samples: int = 6400,
) -> MiniMaxH3ReferenceEncodings:
    """Decode references and encode native conditions without a target/posterior draw.

    The caller validates per-task reference policies and owns frozen encoder
    resources. Image references fit the target area; video references use their
    native downscale-only canvas and24Hz17n+5 extent capped by frame_cap. Text
    samples the same decoded video at2Hz with timestamps0,.5,... . Video audio
    follows its decoded visual duration with exact-window validation; standalone
    audio uses the explicit target-duration sample count and permits short tail
    padding. Preserve reference order and independent audio clock/repair rules.
    Encoders return batch1 FP32 native latents; outputs remove only that axis.
    """
    if audio_decode.sample_rate != 32000 or audio_decode.channels != 2:
        raise ValueError("MiniMaxH3 reference audio requires32kHz stereo")
    if (
        type(frame_cap) is not int
        or frame_cap < 5
        or len(resolution) != 2
        or type(standalone_audio_samples) is not int
        or standalone_audio_samples <= 0
        or any(type(size) is not int or size <= 0 or size % 32 for size in resolution)
        or audio_pad_tolerance < 0
        or audio_max_repair_samples < 0
    ):
        raise ValueError("MiniMaxH3 reference windows require a positive32-grid canvas, frame cap and audio extent")
    visuals, audios, geometries, text_references = [], [], [], []
    for reference in references:
        frames, visual, geometry = None, None, None
        if reference.kind == "image":
            frames = read_reference_image(reference.path, target_size=resolution)
            visual = MiniMaxH3TextVisual(frames)
        elif reference.kind == "video":
            frames = read_reference_video(reference.path, target_frame_count=frame_cap)
            sampled = frames[::12]
            visual = MiniMaxH3TextVisual(sampled, tuple(index / 2 for index in range(len(sampled))))
        if frames is not None:
            latent = condition_video(prepare_pixels(frames))
            if latent.ndim != 5 or latent.shape[:2] != (1, 24) or latent.dtype != torch.float32:
                raise ValueError("MiniMaxH3 reference video encoder must return FP32 B24FHW with batch1")
            visuals.append(latent[0])
            geometry = MiniMaxH3VideoGeometry(*latent.shape[2:])
        audio_frames = 0
        if reference.audio is not None:
            if reference.kind == "video":
                if frames is None:
                    raise ValueError("MiniMaxH3 reference video requires decoded visual frames")
                samples = audio_latent_frames(len(frames)) * 800
            else:
                samples = standalone_audio_samples
            decoded = decode_audio(reference.audio, audio_decode)
            waveform = slice_audio_window(
                decoded.waveform,
                start_sample=0,
                sample_count=samples,
                require_exact=reference.kind == "video",
                pad_tolerance=audio_pad_tolerance,
                repairs=decoded.repairs,
                max_repair_samples=audio_max_repair_samples,
                context=str(reference.path),
            )
            latent = audio(waveform)
            if latent.ndim != 4 or latent.shape[:3] != (1, 32, 2) or latent.dtype != torch.float32:
                raise ValueError("MiniMaxH3 reference audio encoder must return FP32 B32ST with batch1/stereo2")
            audios.append(latent[0])
            audio_frames = latent.shape[-1]
        geometries.append(MiniMaxH3ReferenceGeometry(reference.kind, geometry, audio_frames))
        text_references.append(
            MiniMaxH3TextReference(
                reference.kind,
                visual,
                has_audio=reference.kind == "video" and reference.audio is not None,
            )
        )
    return MiniMaxH3ReferenceEncodings(tuple(visuals), tuple(audios), tuple(geometries), tuple(text_references))
