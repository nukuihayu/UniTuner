"""Stage local native H3 generation resources and atomically publish image or joint AV output."""

import json
from contextlib import closing, nullcontext
from pathlib import Path

import torch

from unituner.adapters.loading import LoadedAdapter
from unituner.artifacts.formats.adapters import adapter_prefixes
from unituner.execution.runtime import runtime_scope
from unituner.models.minimax_h3.config.generation import MiniMaxH3GenerationConfig
from unituner.models.minimax_h3.data.media.output import MiniMaxH3DecodedImage, save_minimax_h3_media
from unituner.models.minimax_h3.inference.inputs import resolve_minimax_h3_inputs
from unituner.models.minimax_h3.inference.loading import (
    load_minimax_h3_render_resources,
    validate_minimax_h3_output,
)
from unituner.models.minimax_h3.inference.sampling import render_minimax_h3
from unituner.models.minimax_h3.loading import load_minimax_h3
from unituner.models.minimax_h3.projections import adapter_projections


@runtime_scope
def run(config: MiniMaxH3GenerationConfig, *, device: str) -> int:
    """Generate one explicit FP32 native H3 result using local artifacts and owned resources.

    Resolve all media before loading models. Build conditions/text without a
    target encode; release the text owner before loading the transformer. VAEs
    stay on CPU until sequential decode. Adapter installation closes on every
    exit; output publication is atomic. Official default BF16 video decode is
    outside this explicit FP32 entry's execution contract.
    """
    selected = torch.device(device)
    model_config, encoders = config.model, config.model.minimax_h3
    if encoders is None or model_config.vae is None or model_config.text_encoder is None:
        raise ValueError("MiniMaxH3 generation requires vae/text_encoder/minimax_h3 artifacts")
    text_device = selected if encoders.text_device is None else torch.device(encoders.text_device)
    if selected.type not in ("cpu", "cuda") or text_device.type not in ("cpu", "cuda"):
        raise ValueError("MiniMaxH3 generation supports CPU/CUDA devices")
    resolved = resolve_minimax_h3_inputs(
        config.inputs, config.sampling, task=model_config.variant, directory=Path.cwd()
    )
    output = Path(config.output).resolve()
    validate_minimax_h3_output(
        output, model_config, (resolved,), adapter=None if config.adapter is None else Path(config.adapter.weights)
    )
    resources = load_minimax_h3_render_resources(model_config, (config,), (resolved,), device=selected)
    model = load_minimax_h3(Path(model_config.weights), device=selected).eval().requires_grad_(False)
    adapter = config.adapter
    if adapter is None:
        owner = nullcontext()
    else:
        targets = adapter_projections(model)
        owner = closing(
            LoadedAdapter(
                Path(adapter.weights),
                adapter.config,
                targets,
                external_prefixes=adapter_prefixes(
                    targets, algorithm=adapter.config.algorithm, namespace="diffusion_model"
                ),
            )
        )
    with owner:
        decoded = render_minimax_h3(
            model, resources.video, config.sampling, resources.inputs[0], audio_vae=resources.audio
        )
        save_minimax_h3_media(decoded, output)
        frames = 1 if isinstance(decoded, MiniMaxH3DecodedImage) else decoded.video.shape[0]
    print(json.dumps({"output": str(output), "frames": frames, "fps": config.sampling.fps, "precision": "fp32"}))
    return 0
