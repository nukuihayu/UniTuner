"""Publish native H3 training previews from owned immutable condition snapshots."""

from pathlib import Path

import torch
from torch import nn

from unitunner.config.training import TrainingConfig
from unitunner.models.minimax_h3.data.media.output import save_minimax_h3_media
from unitunner.models.minimax_h3.inference.inputs import resolve_minimax_h3_inputs
from unitunner.models.minimax_h3.inference.loading import (
    MiniMaxH3RenderResources,
    load_minimax_h3_render_resources,
    validate_minimax_h3_output,
)
from unitunner.models.minimax_h3.inference.sampling import render_minimax_h3
from unitunner.models.minimax_h3.transformer import MiniMaxH3Transformer
from unitunner.precision import sampling_precision
from unitunner.training.previews.context import preview_context


class AudioVideoPreviews:
    """Borrow the live model/adapter and own CPU VAEs plus first-preview condition snapshots.

    Resolve media during construction before training model IO. The first call
    loads/encodes resources inside RNG/mode preservation; later updates reuse
    those immutable input snapshots and sample the current live adapter. No
    text model or GPU VAE remains between previews. The training session owns
    close(); it releases CPU snapshots idempotently without touching the model.
    """

    def __init__(self, config: TrainingConfig) -> None:
        if config.model.family != "minimax-h3" or config.preview is None or not config.preview.audio_video:
            raise ValueError("MiniMaxH3 previews require preview.audio_video")
        self.config = config
        self._resolved = tuple(
            resolve_minimax_h3_inputs(sample.inputs, sample.sampling, task=config.model.variant, directory=Path.cwd())
            for sample in config.preview.audio_video
        )
        self._resources: MiniMaxH3RenderResources | None = None
        self._closed = False

    def __call__(self, update: int, model: MiniMaxH3Transformer, branches: nn.Module) -> None:
        """Publish PNG/MP4 snapshots and restore live flags/RNG even if loading or decoding fails."""
        if self._closed:
            raise RuntimeError("MiniMaxH3 previews are closed")
        if type(update) is not int or update < 0:
            raise ValueError("MiniMaxH3 preview update must be a nonnegative integer")
        config = self.config
        if config.preview is None:
            raise ValueError("MiniMaxH3 previews require configuration")
        samples = config.preview.audio_video
        paths = tuple(
            Path(config.output.directory)
            / "previews"
            / f"step-{update:08d}-{index:02d}{'.png' if sample.sampling.frames == 1 else '.mp4'}"
            for index, sample in enumerate(samples)
        )
        for path in paths:
            validate_minimax_h3_output(path, config.model, self._resolved)
        device = next(model.parameters()).device
        cuda_devices = {device.index or 0} if device.type == "cuda" else set()
        if config.model.minimax_h3 is not None and config.model.minimax_h3.text_device == "cuda":
            cuda_devices.add(torch.cuda.current_device())
        with (
            sampling_precision(config.execution.precision),
            preview_context(model, additional_models=(branches,), cuda_devices=tuple(sorted(cuda_devices))),
        ):
            if self._resources is None:
                self._resources = load_minimax_h3_render_resources(config.model, samples, self._resolved, device=device)
            resources = self._resources
            for sample, inputs, path in zip(samples, resources.inputs, paths, strict=True):
                decoded = render_minimax_h3(
                    model,
                    resources.video,
                    sample.sampling,
                    inputs,
                    audio_vae=resources.audio,
                    allow_cpu_frozen=config.execution.offloads_frozen_weights,
                )
                save_minimax_h3_media(decoded, path)

    def close(self) -> None:
        """Release owned CPU VAEs/condition tensors; subsequent renders reject closed ownership."""
        if not self._closed:
            self._resources = None
            self._closed = True
