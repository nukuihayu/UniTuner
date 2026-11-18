"""Load one owned CPU H3 VAE set and frozen condition snapshots for generation or previews."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from unituner.config.model import ModelConfig
from unituner.models.minimax_h3.config.inputs import MiniMaxH3SampleConfig
from unituner.models.minimax_h3.inference.inputs import MiniMaxH3ResolvedInputs, build_minimax_h3_inputs
from unituner.models.minimax_h3.inference.sampling import MiniMaxH3VideoInputs
from unituner.models.minimax_h3.text.loading import load_text_encoder
from unituner.models.minimax_h3.vae import (
    MiniMaxH3AudioVAE,
    MiniMaxH3VideoVAE,
    load_audio_vae,
    load_video_vae,
)


@dataclass(frozen=True, slots=True)
class MiniMaxH3RenderResources:
    """Own CPU frozen VAEs and condition snapshots; no text model or transformer is retained."""

    video: MiniMaxH3VideoVAE
    audio: MiniMaxH3AudioVAE | None
    inputs: tuple[MiniMaxH3VideoInputs, ...]


def validate_minimax_h3_output(
    output: Path, model: ModelConfig, inputs: Sequence[MiniMaxH3ResolvedInputs], *, adapter: Path | None = None
) -> None:
    """Reject output collisions with every model/adapter/condition across a render group."""
    sources = [Path(model.weights)]
    if model.vae is not None:
        sources.append(Path(model.vae))
    if model.text_encoder is not None:
        sources.append(Path(model.text_encoder))
    if model.minimax_h3 is not None:
        sources.append(Path(model.minimax_h3.audio_vae))
    if adapter is not None:
        sources.append(adapter)
    for request in inputs:
        sources.extend(path for _, path in request.anchors)
        for reference in request.references:
            sources.append(reference.path)
            if reference.audio is not None:
                sources.append(reference.audio.path)
    if any(output.resolve() == path.resolve() for path in sources):
        raise ValueError("MiniMaxH3 output must differ from every model, adapter and condition source")


@torch.no_grad()
def load_minimax_h3_render_resources(
    model: ModelConfig,
    samples: Sequence[MiniMaxH3SampleConfig],
    resolved: Sequence[MiniMaxH3ResolvedInputs],
    *,
    device: torch.device,
) -> MiniMaxH3RenderResources:
    """Load VAEs once, encode all ordered conditions/text, then release the text owner.

    Requests are already path/stream validated before any model IO. Borrow no
    transformer. VAEs remain CPU FP32; video includes its decoder and audio is
    omitted only when all requests are silent single images. Native50layer text
    uses the explicitly configured text device, falling back to device. Returned
    snapshots own CPU tensors and never reread media during subsequent renders.
    """
    encoders = model.minimax_h3
    if model.vae is None or model.text_encoder is None or encoders is None:
        raise ValueError("MiniMaxH3 rendering requires vae/text_encoder/minimax_h3 artifacts")
    if not samples or len(samples) != len(resolved):
        raise ValueError("MiniMaxH3 render samples require matching nonempty resolved inputs")
    text_device = device if encoders.text_device is None else torch.device(encoders.text_device)
    if device.type not in ("cpu", "cuda") or text_device.type not in ("cpu", "cuda"):
        raise ValueError("MiniMaxH3 rendering supports CPU/CUDA devices")
    video = load_video_vae(Path(model.vae), config=encoders.video_topology).eval().requires_grad_(False)
    video_output = any(sample.sampling.frames != 1 for sample in samples)
    audio = None
    if video_output or any(reference.audio is not None for request in resolved for reference in request.references):
        audio = (
            load_audio_vae(Path(encoders.audio_vae), config=encoders.audio_topology, include_decoder=video_output)
            .eval()
            .requires_grad_(False)
        )
    text = load_text_encoder(
        Path(model.text_encoder),
        config_directory=Path(encoders.text_config),
        processor_directory=None if encoders.text_processor is None else Path(encoders.text_processor),
        source_format=model.text_source_format or "auto",
        device=text_device,
        text_config=model.text,
    )
    inputs = tuple(
        build_minimax_h3_inputs(request, sample.sampling, video, text, audio=audio)
        for sample, request in zip(samples, resolved, strict=True)
    )
    return MiniMaxH3RenderResources(video, audio, inputs)
