"""Stage image encoding, native one-step prediction and decoding for image restoration."""

import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch

from unituner.adapters.encoder import load_encoder_adapters
from unituner.artifacts.tensors import TensorReader
from unituner.config.restoration_inference import RestorationInferenceConfig
from unituner.data.media.image import save_image
from unituner.data.media.restoration import read_restoration_image
from unituner.execution.attention import inference_attention
from unituner.execution.runtime import runtime_scope
from unituner.inference.adapters import image_adapter
from unituner.inference.restoration.text import encode_caption
from unituner.inference.restoration.vae import (
    decode_restoration_latents,
    encode_restoration_image,
)
from unituner.models.catalog import ModelFamily
from unituner.models.components.conditioning import Caption
from unituner.models.components.image_encoders import load_image_vae
from unituner.models.flux.conditioning import FluxTextFeatures
from unituner.models.flux.restoration import restore_latents as restore_flux
from unituner.models.flux.transformer import FluxTransformer
from unituner.models.flux.vae import FluxVAE
from unituner.models.flux2.restoration import restore_latents as restore_flux2
from unituner.models.flux2.transformer import Flux2Transformer
from unituner.models.flux2.vae import Flux2VAE
from unituner.models.flux3.restoration import restore_latents as restore_flux3
from unituner.models.flux3.transformer import Flux3Transformer
from unituner.models.image import load_image_model
from unituner.models.krea2.conditioning import Krea2TextFeatures
from unituner.models.krea2.restoration import restore_latents as restore_krea2_latents
from unituner.models.krea2.transformer import Krea2Transformer
from unituner.models.moss_image.restoration import restore_latents as restore_moss
from unituner.models.moss_image.transformer import MossImageTransformer
from unituner.models.qwen_image.restoration import restore_latents as restore_qwen_latents
from unituner.models.qwen_image.transformer import QwenImageTransformer
from unituner.models.qwen_image.vae import QwenImageVAE
from unituner.models.qwen_image_2_1.restoration import restore_latents as restore_qwen21_latents
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unituner.models.qwen_image_2_1.vae import QwenImage21VAE
from unituner.models.z_image.restoration import restore_latents


def _restoration_sigma(config: RestorationInferenceConfig) -> float:
    """Resolve recorded training sigma, rejecting conflicting artifacts or explicit overrides."""
    paths = [config.model.weights]
    if config.adapter is not None:
        paths.append(config.adapter.weights)
    if config.encoder_weights is not None:
        paths.append(config.encoder_weights)
    recorded = []
    for path in paths:
        with TensorReader(Path(path)) as reader:
            value = reader.metadata().get("unituner.restoration.sigma")
        if value is not None:
            try:
                sigma = float(value)
            except ValueError as error:
                raise ValueError("Restoration artifact has invalid sigma metadata") from error
            if not math.isfinite(sigma) or not 0 < sigma < 1:
                raise ValueError("Restoration artifact sigma must be finite and strictly inside (0,1)")
            recorded.append(sigma)
    if recorded and any(value != recorded[0] for value in recorded):
        raise ValueError("Restoration model/adapter/encoder artifacts record conflicting sigma values")
    if config.sigma is not None:
        if recorded and config.sigma != recorded[0]:
            raise ValueError("Explicit restoration sigma differs from the trained artifact")
        return config.sigma
    return recorded[0] if recorded else 0.244


@torch.no_grad()
@runtime_scope
def run(config: RestorationInferenceConfig, *, device: str) -> int:
    """Restore one RGB image with FP32 posterior-mean encoding and direct one-step prediction.

    Use local artifacts only. Text, encoder, denoiser and decoder occupy the
    selected device in stages. Preserve oriented input dimensions via reflection
    padding/cropping. Optional native encoder and published denoiser adapters have
    scoped reversible owners. Atomically publish PNG only after successful decode.
    """
    with inference_attention(config.attention_backend):
        family = config.model.family
        flux = family == ModelFamily.FLUX
        qwen = config.model.family == ModelFamily.QWEN_IMAGE
        krea = config.model.family == ModelFamily.KREA2
        if config.model.family not in (
            ModelFamily.Z_IMAGE,
            ModelFamily.QWEN_IMAGE,
            ModelFamily.QWEN_IMAGE_2_1,
            ModelFamily.KREA2,
            ModelFamily.FLUX,
            ModelFamily.FLUX2,
            ModelFamily.FLUX3,
            ModelFamily.MOSS_IMAGE,
        ):
            raise ValueError("Restoration entry supports Z-Image, original Qwen Image, Qwen Image2.1 or Krea2")
        if qwen and config.model.variant != "original":
            raise ValueError("Qwen restoration requires original variant")
        if flux and config.model.variant != "dev":
            raise ValueError("FLUX restoration requires the text-to-image dev variant")
        vae_path, text_path = config.model.vae, config.model.text_encoder
        if vae_path is None or text_path is None:
            raise ValueError("Restoration requires model.vae and model.text_encoder")
        if Path(config.input).resolve() == Path(config.output).resolve():
            raise ValueError("Restoration output must differ from the input image")
        selected = torch.device(device)
        if selected.type not in {"cpu", "cuda"}:
            raise ValueError("Restoration entry supports CPU and CUDA devices")
        sigma = _restoration_sigma(config)
        vae = load_image_vae(Path(vae_path), family=config.model.family, device=torch.device("cpu"))
        if isinstance(vae, QwenImageVAE) and vae.config.input_channels != 3:
            raise ValueError("Qwen restoration requires RGB vae")
        if krea and (
            vae.config.latent_channels != 16 or not isinstance(vae, QwenImageVAE) or vae.config.spatial_factor != 8
        ):
            raise ValueError("Krea2 restoration requires shared RGB Qwen vae with latent16/factor8")
        factor = (
            vae.config.spatial_factor
            if isinstance(vae, QwenImageVAE | QwenImage21VAE | FluxVAE | Flux2VAE)
            else 2 ** (len(vae.config.block_channels) - 1)
        )
        pixels, (width, height) = read_restoration_image(Path(config.input), granularity=2 * factor)
        caption = encode_caption(config, device=selected)
        vae.to(selected)
        with ExitStack() as stack:
            stack.enter_context(torch.autocast(selected.type, enabled=False))
            encoder = None
            if config.encoder_weights is not None:
                encoder = load_encoder_adapters(vae.encoder, Path(config.encoder_weights))
                stack.callback(encoder.close)
            low_latents = encode_restoration_image(vae, pixels.unsqueeze(0).to(selected)).cpu()
        if config.encoder_weights is not None:
            del encoder
        vae.cpu()
        model = load_image_model(config.model, device=selected, dtype=torch.float32).eval().requires_grad_(False)
        channels = (
            model.config.out_channels // 4
            if isinstance(model, FluxTransformer)
            else model.config.image_channels
            if isinstance(model, Flux3Transformer)
            else model.config.latent_channels
            if isinstance(model, MossImageTransformer)
            else model.config.out_channels
            if isinstance(model, QwenImageTransformer | QwenImage21Transformer)
            else model.config.in_channels
        )
        if (
            not isinstance(model, QwenImage21Transformer) and model.config.variant != config.model.variant
        ) or channels != vae.config.latent_channels * (4 if isinstance(vae, Flux2VAE) else 1):
            raise ValueError("Restoration model variant/channels differ from the supplied model/vae")
        with torch.autocast(selected.type, enabled=False), image_adapter(model, config.adapter):
            if isinstance(model, FluxTransformer):
                if not isinstance(caption, FluxTextFeatures):
                    raise ValueError("FLUX requires pooled CLIP and T5 features")
                restored = restore_flux(
                    model,
                    low_latents.to(selected),
                    FluxTextFeatures(caption.captions.to(selected), caption.pooled.to(selected)),
                    sigma=sigma,
                ).cpu()
            elif isinstance(model, Krea2Transformer):
                if not isinstance(caption, Krea2TextFeatures):
                    raise ValueError("Krea2 restoration requires selected-layer captions")
                restored = restore_krea2_latents(
                    model,
                    low_latents.to(selected),
                    Krea2TextFeatures(caption.features.to(selected), caption.lengths),
                    sigma=sigma,
                ).cpu()
            else:
                if not isinstance(caption, Caption):
                    raise ValueError("Selected-layer captions require Krea2 restoration")
                condition = Caption(
                    caption.features.to(device=selected, dtype=low_latents.dtype),
                    None if caption.mask is None else caption.mask.to(selected),
                    caption.lengths,
                )
                restored = (
                    restore_flux2(model, low_latents.to(selected), condition, sigma=sigma)
                    if isinstance(model, Flux2Transformer)
                    else restore_flux3(model, low_latents.to(selected), condition, sigma=sigma)
                    if isinstance(model, Flux3Transformer)
                    else restore_moss(model, low_latents.to(selected), condition, sigma=sigma)
                    if isinstance(model, MossImageTransformer)
                    else restore_qwen21_latents(model, low_latents.to(selected), condition, sigma=sigma)
                    if isinstance(model, QwenImage21Transformer)
                    else restore_qwen_latents(model, low_latents.to(selected), condition, sigma=sigma)
                    if isinstance(model, QwenImageTransformer)
                    else restore_latents(model, low_latents.to(selected), condition, sigma=sigma)
                ).cpu()
        del model
        vae.to(selected)
        with torch.autocast(selected.type, enabled=False):
            result = decode_restoration_latents(vae, restored.to(selected))[0, :, :height, :width]
        save_image(result, Path(config.output))
        print(json.dumps({"restored": True, "width": width, "height": height}))
        return 0
