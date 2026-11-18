"""Bind native FLUX.2 packed-VAE flow math to shared parameter/execution owners."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from unituner.execution.regions import ModelRegion, ModelRegions
from unituner.models.components.conditioning import Caption
from unituner.models.flux2.conditioning import build_image_inputs, pack_text, unpack_image
from unituner.models.flux2.projections import adapter_projections
from unituner.models.flux2.restoration import restore_latents
from unituner.models.flux2.transformer import Flux2Transformer
from unituner.training.contracts import ImageFlowPrediction, StepResult
from unituner.training.losses.wavelet import WaveletLoss, image_flow_loss


@dataclass(frozen=True, slots=True)
class Flux2FlowRequest:
    """Carry BCHW clean/noise, B noise fractions, per-sample LD text and CHW references.

    Latents already include vae2x2 packing and BN normalization. Each sample's
    clean reference images may have independent geometry; no padding is attended.
    Dev uses embedded guidance1 by default during native flow finetuning.
    """

    latents: Tensor
    noise: Tensor
    sigma: Tensor
    captions: tuple[Tensor, ...]
    references: tuple[tuple[Tensor, ...], ...] = ()
    guidance: float = 1.0


class Flux2Binding:
    """Borrow native double/single regions for shared full/adapter execution."""

    def __init__(self, model: Flux2Transformer, *, wavelet: WaveletLoss | None = None) -> None:
        self.model = model
        self.wavelet = wavelet
        blocks = tuple(
            ModelRegion(block, index, "visual")
            for index, block in enumerate((*model.double_blocks, *model.single_blocks))
        )
        self.regions = ModelRegions(blocks)

    def adapter_targets(self) -> dict[str, nn.Linear]:
        """Borrow native attention/FFN projections, excluding shared modulation."""
        return adapter_projections(self.model)

    def restore(self, low_latents: Tensor, captions: Caption, *, sigma: float) -> Tensor:
        """Restore native latents with the direct mid-timestep Euler method."""
        return restore_latents(self.model, low_latents, captions, sigma=sigma)

    def predict(self, request: Flux2FlowRequest) -> ImageFlowPrediction:
        """Return BCHW velocity with native unrounded sigma and noise-minus-clean target.

        Interpolate before row-major packing; append ordered clean references and
        trim their predictions. Retain every caption token, including padding.
        Inputs and gradients remain owned by the caller; no RNG or mode mutation.
        """
        clean, noise, sigma = request.latents, request.noise, request.sigma
        if (
            clean.ndim != 4
            or min(clean.shape) < 1
            or noise.shape != clean.shape
            or noise.dtype != clean.dtype
            or noise.device != clean.device
            or len(request.captions) != clean.shape[0]
            or (request.references and len(request.references) != clean.shape[0])
        ):
            raise ValueError("FLUX.2 requires matching BCHW clean/noise and one caption/reference row per sample")
        if (
            sigma.shape != (clean.shape[0],)
            or sigma.device != clean.device
            or not sigma.is_floating_point()
            or not torch.all(torch.isfinite(sigma) & (sigma >= 0) & (sigma <= 1))
        ):
            raise ValueError("FLUX.2 sigma must be a finite B vector in[0,1] on the latent device")
        if not math.isfinite(request.guidance) or request.guidance < 0:
            raise ValueError("FLUX.2 training guidance must be finite and nonnegative")
        if not self.model.config.use_guidance_embed and request.guidance != 1.0:
            raise ValueError("FLUX.2 Klein training does not consume embedded guidance")
        fraction = sigma.to(clean.dtype)[:, None, None, None]
        noisy = (1 - fraction) * clean + fraction * noise
        rows = []
        for index, target in enumerate(noisy):
            image = build_image_inputs(target, request.references[index] if request.references else ())
            captions, ids = pack_text(request.captions[index])
            guidance = target.new_full((1,), request.guidance) if self.model.config.use_guidance_embed else None
            prediction = self.model(
                image.tokens,
                image.ids,
                sigma[index : index + 1],
                captions.unsqueeze(0),
                ids.unsqueeze(0),
                guidance,
                target_count=image.target_count,
            )
            rows.append(unpack_image(prediction[0, : image.target_count], height=clean.shape[2], width=clean.shape[3]))
        return ImageFlowPrediction(torch.stack(rows), noise - clean, sigma)

    def step(self, request: Flux2FlowRequest) -> StepResult:
        """Return FP32 element-mean native flow MSE with the actual sample count."""
        result = self.predict(request)
        return StepResult(
            image_flow_loss(result.prediction, result.target, request.sigma, self.wavelet), request.latents.shape[0]
        )
