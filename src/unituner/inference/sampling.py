"""Dispatch image sampling by explicit native model and text contracts."""

import torch
from torch import Tensor

from unituner.config.sampling import SamplingConfig
from unituner.execution.attention import inference_attention
from unituner.models.components.conditioning import Caption
from unituner.models.flux.conditioning import FluxCondition, FluxTextFeatures
from unituner.models.flux.inference import sampling as flux_sampling
from unituner.models.flux.transformer import FluxTransformer
from unituner.models.flux2.inference import sampling as flux2_sampling
from unituner.models.flux2.transformer import Flux2Transformer
from unituner.models.flux3.inference import sampling as flux3_sampling
from unituner.models.flux3.transformer import Flux3Transformer
from unituner.models.image import ImageModel
from unituner.models.krea2.conditioning import Krea2TextFeatures
from unituner.models.krea2.inference import sampling as krea2_sampling
from unituner.models.krea2.transformer import Krea2Transformer
from unituner.models.moss_image.inference import sampling as moss_image_sampling
from unituner.models.moss_image.transformer import MossImageTransformer
from unituner.models.qwen_image.inference import sampling as qwen_image_sampling
from unituner.models.qwen_image.transformer import QwenImageTransformer
from unituner.models.qwen_image_2_1.conditioning import QwenImage21TextFeatures
from unituner.models.qwen_image_2_1.inference import sampling as qwen_image_2_1_sampling
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unituner.models.z_image.inference import sampling as z_image_sampling
from unituner.models.z_image.transformer import ZImageTransformer


@torch.no_grad()
def generate_latents(
    model: ImageModel,
    positive: Caption | FluxTextFeatures | QwenImage21TextFeatures | Krea2TextFeatures,
    sampling: SamplingConfig,
    *,
    negative: Caption | QwenImage21TextFeatures | Krea2TextFeatures | None = None,
    vae_factor: int | None = None,
    condition: FluxCondition | None = None,
    references: tuple[Tensor, ...] = (),
    inpainting_mask: Tensor | None = None,
    allow_cpu_frozen: bool = False,
) -> Tensor:
    """Dispatch native noise initialization and sampling without owning models or RNG.

    The model family selects its feature contract and supported conditions. Explicit
    unsupported inputs raise; no family falls back to another model implementation."""
    with inference_attention(sampling.attention_backend):
        if isinstance(model, FluxTransformer):
            if not isinstance(positive, FluxTextFeatures) or (
                negative is not None and (not isinstance(negative, Caption))
            ):
                raise ValueError("flux requires native FluxTextFeatures text features")
            return flux_sampling.generate_latents(
                model,
                positive,
                sampling,
                allow_cpu_frozen=allow_cpu_frozen,
                negative=negative,
                vae_factor=vae_factor,
                condition=condition,
                references=references,
                inpainting_mask=inpainting_mask,
            )
        if isinstance(model, Flux2Transformer):
            if not isinstance(positive, Caption) or (negative is not None and (not isinstance(negative, Caption))):
                raise ValueError("flux2 requires native Caption text features")
            return flux2_sampling.generate_latents(
                model,
                positive,
                sampling,
                allow_cpu_frozen=allow_cpu_frozen,
                negative=negative,
                vae_factor=vae_factor,
                condition=condition,
                references=references,
                inpainting_mask=inpainting_mask,
            )
        if isinstance(model, Flux3Transformer):
            if not isinstance(positive, Caption) or (negative is not None and (not isinstance(negative, Caption))):
                raise ValueError("flux3 requires native Caption text features")
            return flux3_sampling.generate_latents(
                model,
                positive,
                sampling,
                allow_cpu_frozen=allow_cpu_frozen,
                negative=negative,
                vae_factor=vae_factor,
                condition=condition,
                references=references,
                inpainting_mask=inpainting_mask,
            )
        if isinstance(model, MossImageTransformer):
            if not isinstance(positive, Caption) or (negative is not None and (not isinstance(negative, Caption))):
                raise ValueError("moss_image requires native Caption text features")
            return moss_image_sampling.generate_latents(
                model,
                positive,
                sampling,
                negative=negative,
                vae_factor=vae_factor,
                condition=condition,
                references=references,
                inpainting_mask=inpainting_mask,
            )
        if isinstance(model, ZImageTransformer):
            if not isinstance(positive, Caption) or (negative is not None and (not isinstance(negative, Caption))):
                raise ValueError("z_image requires native Caption text features")
            return z_image_sampling.generate_latents(
                model,
                positive,
                sampling,
                negative=negative,
                vae_factor=vae_factor,
                condition=condition,
                references=references,
                inpainting_mask=inpainting_mask,
            )
        if isinstance(model, QwenImageTransformer):
            if not isinstance(positive, Caption) or (negative is not None and (not isinstance(negative, Caption))):
                raise ValueError("qwen_image requires native Caption text features")
            return qwen_image_sampling.generate_latents(
                model,
                positive,
                sampling,
                allow_cpu_frozen=allow_cpu_frozen,
                negative=negative,
                vae_factor=vae_factor,
                condition=condition,
                references=references,
                inpainting_mask=inpainting_mask,
            )
        if isinstance(model, QwenImage21Transformer):
            if not isinstance(positive, QwenImage21TextFeatures) or (
                negative is not None and (not isinstance(negative, QwenImage21TextFeatures))
            ):
                raise ValueError("qwen_image_2_1 requires native QwenImage21TextFeatures text features")
            return qwen_image_2_1_sampling.generate_latents(
                model,
                positive,
                sampling,
                negative=negative,
                vae_factor=vae_factor,
                condition=condition,
                references=references,
                inpainting_mask=inpainting_mask,
            )
        if isinstance(model, Krea2Transformer):
            if not isinstance(positive, Krea2TextFeatures) or (
                negative is not None and (not isinstance(negative, Krea2TextFeatures))
            ):
                raise ValueError("krea2 requires native Krea2TextFeatures text features")
            return krea2_sampling.generate_latents(
                model,
                positive,
                sampling,
                negative=negative,
                vae_factor=vae_factor,
                condition=condition,
                references=references,
                inpainting_mask=inpainting_mask,
            )
        raise TypeError("No image sampler for this model type")
