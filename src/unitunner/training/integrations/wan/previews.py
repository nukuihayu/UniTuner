"""Publish native Wan training previews using the shared video inference pipeline."""

from pathlib import Path

from torch import nn

from unitunner.config.training import TrainingConfig
from unitunner.data.media.video_output import save_video
from unitunner.models.qwen_image.loading import load_qwen_image_vae
from unitunner.models.wan.inference.inputs import build_wan_video_inputs
from unitunner.models.wan.inference.sampling import render_wan_video
from unitunner.models.wan.transformer import WanTransformer
from unitunner.training.previews.context import preview_context


class VideoPreviews:
    """Borrow the active expert/adapter and own no resources outside each preview context.

    Asset loading, local-seed sampling, VAE decode and media IO occur entirely
    inside RNG/mode preservation. Reuse the public video inference functions;
    optimizer state, training data cursor and checkpoint identity are untouched.
    """

    def __init__(self, config: TrainingConfig) -> None:
        if config.model.family != "wan" or config.preview is None or not config.preview.videos:
            raise ValueError("Wan previews require preview.videos")
        self.config = config

    def __call__(self, update: int, model: WanTransformer, branches: nn.Module) -> None:
        """Publish one MP4 per configured prompt with exception-safe training restoration."""
        if type(update) is not int or update < 0:
            raise ValueError("Preview update must be a nonnegative integer")
        config = self.config
        if config.preview is None or config.model.vae is None:
            raise ValueError("Wan previews require vae and preview configuration")
        device = next(model.parameters()).device
        with preview_context(
            model, additional_models=(branches,), cuda_devices=(device.index or 0,) if device.type == "cuda" else ()
        ):
            vae = load_qwen_image_vae(Path(config.model.vae)).eval().requires_grad_(False)
            for index, sampling in enumerate(config.preview.videos):
                inputs = build_wan_video_inputs(config.model, sampling, vae, text_length=model.config.text_length)
                pixels = render_wan_video(
                    model,
                    vae,
                    sampling,
                    inputs,
                    allow_cpu_frozen=self.config.execution.offloads_frozen_weights,
                )
                save_video(
                    pixels,
                    Path(config.output.directory) / "previews" / f"step-{update:08d}-{index:02d}.mp4",
                    fps=sampling.fps,
                )
