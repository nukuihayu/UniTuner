"""Stage local native Wan resources and publish one generated video."""

import json
from contextlib import ExitStack, closing
from pathlib import Path

import torch

from unitunner.adapters.loading import LoadedAdapter
from unitunner.artifacts.formats.adapters import adapter_prefixes
from unitunner.data.media.video_output import save_video
from unitunner.execution.runtime import runtime_scope
from unitunner.models.qwen_image.loading import load_qwen_image_vae
from unitunner.models.wan.config.generation import WanGenerationConfig
from unitunner.models.wan.inference.inputs import build_wan_video_inputs
from unitunner.models.wan.inference.sampling import render_wan_video
from unitunner.models.wan.loading import load_wan
from unitunner.models.wan.projections import adapter_projections


@runtime_scope
def run(config: WanGenerationConfig, *, device: str) -> int:
    """Generate one resident FP32 Wan video, closing optional adapters on every exit."""
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("Wan video generation supports CPU/CUDA only")
    if config.model.vae is None:
        raise ValueError("Wan video generation requires vae")
    if (
        config.sampling.control_video is not None
        and Path(config.output).resolve() == Path(config.sampling.control_video).resolve()
    ):
        raise ValueError("Wan video output must differ from its control source")
    vae = load_qwen_image_vae(Path(config.model.vae)).eval().requires_grad_(False)
    model = (
        load_wan(
            Path(config.model.weights), variant=config.model.variant, version=config.model.version, device=selected
        )
        .eval()
        .requires_grad_(False)
    )
    if model.config.version != config.model.version or model.config.variant != config.model.variant:
        raise ValueError("Wan generation model topology differs from configured version/variant")
    inputs = build_wan_video_inputs(config.model, config.sampling, vae, text_length=model.config.text_length)
    high = (
        None
        if config.high_noise is None
        else load_wan(
            Path(config.high_noise.weights), variant=config.model.variant, version=config.model.version, device=selected
        )
        .eval()
        .requires_grad_(False)
    )
    with ExitStack() as owners:
        experts = [(model, config.adapter)]
        if high is not None and config.high_noise is not None:
            experts.append((high, config.high_noise.adapter))
        for expert, adapter in experts:
            if expert is not None and adapter is not None:
                targets = adapter_projections(expert)
                owners.enter_context(
                    closing(
                        LoadedAdapter(
                            Path(adapter.weights),
                            adapter.config,
                            targets,
                            external_prefixes=adapter_prefixes(
                                targets, algorithm=adapter.config.algorithm, namespace="diffusion_model"
                            ),
                        )
                    )
                )
        pixels = render_wan_video(
            model,
            vae,
            config.sampling,
            inputs,
            high_noise_model=high,
            boundary=None if config.high_noise is None else config.high_noise.boundary,
            high_noise_guidance=None if config.high_noise is None else config.high_noise.guidance,
        )
        save_video(pixels, Path(config.output), fps=config.sampling.fps)
    print(json.dumps({"output": config.output, "frames": config.sampling.frames, "fps": config.sampling.fps}))
    return 0
