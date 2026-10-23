"""Stage image encoding, native one-step prediction and decoding for image restoration."""

import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch

from unitunner.adapters.encoder import load_encoder_adapters
from unitunner.adapters.loading import LoadedAdapter
from unitunner.artifacts.formats.adapters import adapter_prefixes
from unitunner.artifacts.formats.flux_adapters import flux_adapter_layout
from unitunner.artifacts.formats.krea2_adapters import krea2_adapter_prefixes
from unitunner.artifacts.tensors import TensorReader
from unitunner.config.restoration_inference import RestorationInferenceConfig
from unitunner.data.media.image import save_image
from unitunner.data.media.restoration import read_restoration_image
from unitunner.execution.attention import inference_attention
from unitunner.execution.runtime import runtime_scope
from unitunner.inference.restoration.text import encode_caption
from unitunner.inference.restoration.vae import (
    decode_restoration_latents,
    encode_restoration_image,
    load_restoration_vae,
)
from unitunner.models.catalog import ModelFamily
from unitunner.models.components.conditioning import Caption
from unitunner.models.flux.conditioning import FluxTextFeatures
from unitunner.models.flux.loading import load_flux
from unitunner.models.flux.projections import adapter_projections as flux_projections
from unitunner.models.flux.restoration import restore_latents as restore_flux
from unitunner.models.flux.transformer import FluxTransformer
from unitunner.models.flux.vae import FluxVAE
from unitunner.models.flux2.loading import load_flux2
from unitunner.models.flux2.projections import adapter_projections as flux2_projections
from unitunner.models.flux2.restoration import restore_latents as restore_flux2
from unitunner.models.flux2.transformer import Flux2Transformer
from unitunner.models.flux2.vae import Flux2VAE
from unitunner.models.flux3.loading import load_flux3
from unitunner.models.flux3.projections import adapter_projections as flux3_projections
from unitunner.models.flux3.restoration import restore_latents as restore_flux3
from unitunner.models.flux3.transformer import Flux3Transformer
from unitunner.models.krea2.conditioning import Krea2TextFeatures
from unitunner.models.krea2.loading import load_krea2
from unitunner.models.krea2.projections import adapter_projections as krea2_adapter_projections
from unitunner.models.krea2.restoration import restore_latents as restore_krea2_latents
from unitunner.models.krea2.transformer import Krea2Transformer
from unitunner.models.moss_image.loading import load_moss_image
from unitunner.models.moss_image.projections import adapter_projections as moss_projections
from unitunner.models.moss_image.restoration import restore_latents as restore_moss
from unitunner.models.moss_image.transformer import MossImageTransformer
from unitunner.models.qwen_image.loading import load_qwen_image
from unitunner.models.qwen_image.projections import adapter_projections as qwen_adapter_projections
from unitunner.models.qwen_image.restoration import restore_latents as restore_qwen_latents
from unitunner.models.qwen_image.transformer import QwenImageTransformer
from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.models.qwen_image_2_1.loading import load_qwen_image_21
from unitunner.models.qwen_image_2_1.projections import adapter_projections as qwen21_adapter_projections
from unitunner.models.qwen_image_2_1.restoration import restore_latents as restore_qwen21_latents
from unitunner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unitunner.models.qwen_image_2_1.vae import QwenImage21VAE
from unitunner.models.z_image.loading import load_z_image
from unitunner.models.z_image.projections import adapter_projections
from unitunner.models.z_image.restoration import restore_latents


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
            value = reader.metadata().get("unitunner.restoration.sigma")
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
        flux, flux2, flux3, moss = (
            family == item for item in (ModelFamily.FLUX, ModelFamily.FLUX2, ModelFamily.FLUX3, ModelFamily.MOSS_IMAGE)
        )
        qwen = config.model.family == ModelFamily.QWEN_IMAGE
        qwen21 = config.model.family == ModelFamily.QWEN_IMAGE_2_1
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
        vae = load_restoration_vae(Path(vae_path), family=config.model.family, device=torch.device("cpu"))
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
        adapter = None
        targets = None
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
        model = (
            (
                load_flux(Path(config.model.weights), variant=config.model.variant, device=selected)
                if flux
                else load_flux2(
                    Path(config.model.weights),
                    variant=config.model.variant,
                    version=config.model.version,
                    device=selected,
                )
                if flux2
                else load_flux3(Path(config.model.weights), variant=config.model.variant, device=selected)
                if flux3
                else load_moss_image(Path(config.model.weights), variant=config.model.variant, device=selected)
                if moss
                else load_krea2(
                    Path(config.model.weights), variant=config.model.variant, device=selected, dtype=torch.float32
                )
                if krea
                else load_qwen_image_21(Path(config.model.weights), device=selected, dtype=torch.float32)
                if qwen21
                else load_qwen_image(
                    Path(config.model.weights), variant=config.model.variant, device=selected, dtype=torch.float32
                )
                if qwen
                else load_z_image(
                    Path(config.model.weights), variant=config.model.variant, device=selected, dtype=torch.float32
                )
            )
            .eval()
            .requires_grad_(False)
        )
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
        adapter = None
        targets = None
        with ExitStack() as stack:
            stack.enter_context(torch.autocast(selected.type, enabled=False))
            if config.adapter is not None:
                targets = (
                    flux_projections(model)
                    if isinstance(model, FluxTransformer)
                    else flux2_projections(model)
                    if isinstance(model, Flux2Transformer)
                    else flux3_projections(model)
                    if isinstance(model, Flux3Transformer)
                    else moss_projections(model)
                    if isinstance(model, MossImageTransformer)
                    else krea2_adapter_projections(model)
                    if isinstance(model, Krea2Transformer)
                    else qwen21_adapter_projections(model)
                    if isinstance(model, QwenImage21Transformer)
                    else qwen_adapter_projections(model)
                    if isinstance(model, QwenImageTransformer)
                    else adapter_projections(model)
                )
                adapter = LoadedAdapter(
                    Path(config.adapter.weights),
                    config.adapter.config,
                    targets,
                    projection_layout=flux_adapter_layout(
                        targets, hidden_size=model.config.hidden_size, algorithm=config.adapter.config.algorithm
                    )
                    if isinstance(model, FluxTransformer | Flux2Transformer)
                    else None,
                    external_prefixes=None
                    if isinstance(model, FluxTransformer | Flux2Transformer)
                    else (
                        krea2_adapter_prefixes(targets, algorithm=config.adapter.config.algorithm)
                        if isinstance(model, Krea2Transformer)
                        else adapter_prefixes(
                            targets,
                            algorithm=config.adapter.config.algorithm,
                            namespace="diffusion_model"
                            if isinstance(model, Flux3Transformer | MossImageTransformer)
                            else "transformer",
                        )
                    ),
                )
                stack.callback(adapter.close)
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
        if config.adapter is not None:
            del adapter, targets
        del model
        vae.to(selected)
        with torch.autocast(selected.type, enabled=False):
            result = decode_restoration_latents(vae, restored.to(selected))[0, :, :height, :width]
        save_image(result, Path(config.output))
        print(json.dumps({"restored": True, "width": width, "height": height}))
        return 0
