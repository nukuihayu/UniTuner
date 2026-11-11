"""Native H3-Image generate/restore providers using shared image command owners."""

import json
from contextlib import ExitStack, closing
from pathlib import Path

import torch

from unituner.adapters.encoder import load_encoder_adapters
from unituner.config.generation import GenerationConfig
from unituner.config.model import ModelConfig
from unituner.config.restoration_inference import RestorationInferenceConfig
from unituner.data.media.image import save_image
from unituner.data.media.restoration import read_restoration_image
from unituner.execution.attention import inference_attention
from unituner.execution.precision import artifact_precision
from unituner.execution.runtime import runtime_scope
from unituner.inference.adapters import image_adapter
from unituner.inference.publication import publish_images
from unituner.inference.restoration.sigma import restoration_sigma
from unituner.inference.restoration.vae import encode_restoration_image
from unituner.models.minimax_h3.config.model import MiniMaxH3Config
from unituner.models.minimax_h3.image import load_image_transformer
from unituner.models.minimax_h3.image_encoders import load_image_encoders
from unituner.models.minimax_h3.image_inference import encode_image_condition, generate_image
from unituner.models.minimax_h3.projections import adapter_projections
from unituner.precision import FLOAT_DTYPES
from unituner.training.integrations.minimax_h3.image import MiniMaxH3ImageBinding


def _model(config: ModelConfig, device: torch.device):
    if config.pdd_config is None:
        raise ValueError("MiniMax-H3-Image needs model.pdd_config")
    return (
        load_image_transformer(
            Path(config.weights),
            config=config.image_topology or MiniMaxH3Config(),
            pdd_config=Path(config.pdd_config),
            pdd_heads=None if config.pdd_heads is None else Path(config.pdd_heads),
            device=device,
            dtype=FLOAT_DTYPES[artifact_precision(Path(config.weights))],
        )
        .eval()
        .requires_grad_(False)
    )


@torch.no_grad()
@runtime_scope
def generate(config: GenerationConfig, *, device: torch.device) -> int:
    """Generate a PNG through the published six-step PDD grid and ordered edit images."""
    width = config.sampling.width if "width" in config.sampling.model_fields_set else None
    height = config.sampling.height if "height" in config.sampling.model_fields_set else None
    with inference_attention(config.sampling.attention_backend):
        vae, text = load_image_encoders(config.model, device=torch.device("cpu"))
        model = _model(config.model, device)
        with image_adapter(model, config.adapter, targets=adapter_projections(model)):
            pixels = generate_image(
                model,
                vae,
                text,
                config.sampling.prompt,
                width=width,
                height=height,
                size_policy=config.sampling.h3_size_policy,
                seed=config.sampling.seed,
                images=tuple(Path(path) for path in config.sampling.reference_images),
            )
    return publish_images(pixels[:, :, 0], Path(config.output), rounding="nearest")


@torch.no_grad()
@runtime_scope
def restore(config: RestorationInferenceConfig, *, device: str) -> int:
    """Restore a native H3-Image input with the recorded one-step sigma."""
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("H3-Image restoration requires CPU/CUDA")
    if Path(config.input).resolve() == Path(config.output).resolve():
        raise ValueError("Restoration output must differ from input")
    sigma = restoration_sigma(config)
    with inference_attention(config.attention_backend):
        vae, text_encoder = load_image_encoders(config.model, device=selected, text_device=torch.device("cpu"))
        pixels, (width, height) = read_restoration_image(Path(config.input), granularity=32)
        text, tags, references, _ = encode_image_condition(
            vae,
            text_encoder,
            config.prompt,
            width=pixels.shape[-1],
            height=pixels.shape[-2],
            images=tuple(Path(path) for path in config.reference_images),
            model_device=selected,
        )
        with ExitStack() as stack:
            if config.encoder_weights is not None:
                stack.enter_context(closing(load_encoder_adapters(vae.encoder, Path(config.encoder_weights))))
            low = encode_restoration_image(vae, pixels.unsqueeze(0).to(selected), posterior="mean")
        model = _model(config.model, selected)
        with image_adapter(model, config.adapter, targets=adapter_projections(model)):
            result = MiniMaxH3ImageBinding(model).restore(
                low,
                text,
                tags,
                sigma=sigma,
                references=references,
            )
        restored = vae.decode(result)[:, :, 0][0, :, :height, :width].float().cpu()
        save_image(restored, Path(config.output))
    print(json.dumps({"restored": True, "width": width, "height": height}))
    return 0
