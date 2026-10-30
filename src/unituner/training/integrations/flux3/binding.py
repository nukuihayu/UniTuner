"""Bind native FLUX3 three-mode flow predictions to shared parameter/execution owners."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from unituner.execution.regions import ModelRegion, ModelRegions, RegionBranch
from unituner.models.components.conditioning import Caption
from unituner.models.flux3.conditioning import (
    Flux3Mode,
    Flux3Sample,
    image_tokens,
    pack_samples,
    reference_tokens,
    text_tokens,
)
from unituner.models.flux3.projections import adapter_projections
from unituner.models.flux3.restoration import restore_latents
from unituner.models.flux3.transformer import Flux3Transformer
from unituner.training.contracts import ImageFlowPrediction, StepResult
from unituner.training.losses.wavelet import WaveletLoss, image_flow_loss


@dataclass(frozen=True, slots=True)
class Flux3FlowRequest:
    """Carry BCHW packed-VAE clean/noise, B sigma, per-sample LD text and references.

    Unlike FLUX.1, the VAE has already packed 2x2 cells into channels. References
    are ordered tuples of CHW normalized VAE latents. The distilled flow-map
    finetuning destination is the current sigma, independently of inference steps.
    """

    latents: Tensor
    noise: Tensor
    sigma: Tensor
    captions: tuple[Tensor, ...]
    references: tuple[tuple[Tensor, ...], ...] = ()


class Flux3Binding:
    """Borrow native branch/shared regions without attaching policy attributes to models."""

    def __init__(self, model: Flux3Transformer, *, wavelet: WaveletLoss | None = None) -> None:
        self.model = model
        self.wavelet = wavelet
        # Match the transformer's depth-major mode traversal. Logical indices
        # remain branch-local; residency lookahead follows actual invocation order.
        modes: tuple[tuple[Flux3Mode, RegionBranch], ...] = (
            ("image", "visual"),
            ("image_cond", "condition"),
            ("text", "text"),
        )
        blocks = tuple(
            ModelRegion(model.early_blocks(mode)[depth], depth, branch)
            for depth in range(model.config.depth)
            for mode, branch in modes
        ) + tuple(
            ModelRegion(block, model.config.depth + index, "visual") for index, block in enumerate(model.single_blocks)
        )
        self.regions = ModelRegions(blocks)

    def adapter_targets(self) -> dict[str, nn.Linear]:
        """Borrow native attention/FFN projections for all three modes and shared blocks."""
        return adapter_projections(self.model)

    def restore(self, low_latents: Tensor, captions: Caption, *, sigma: float) -> Tensor:
        """Restore native latents with the direct mid-timestep Euler method."""
        return restore_latents(self.model, low_latents, captions, sigma=sigma)

    def predict(self, request: Flux3FlowRequest) -> ImageFlowPrediction:
        """Return BCHW noise-minus-clean flow without rounding sigma or dropping text tokens."""
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
            raise ValueError("FLUX3 requires matching BCHW clean/noise and one text/reference row per sample")
        if (
            sigma.shape != (clean.shape[0],)
            or sigma.device != clean.device
            or not sigma.is_floating_point()
            or not torch.all(torch.isfinite(sigma) & (sigma >= 0) & (sigma <= 1))
        ):
            raise ValueError("FLUX3 sigma must be a finite B vector in [0,1] on the latent device")
        fraction = sigma.to(clean.dtype)[:, None, None, None]
        noisy = (1 - fraction) * clean + fraction * noise
        samples = [
            Flux3Sample(
                image_tokens(value),
                text_tokens(request.captions[index]),
                reference_tokens(request.references[index]) if request.references else None,
            )
            for index, value in enumerate(noisy)
        ]
        packed = pack_samples(samples, sigma)
        prediction = self.model(packed, sigma if self.model.config.variant == "distilled" else None)
        prediction = prediction.reshape(clean.shape[0], clean.shape[2], clean.shape[3], clean.shape[1]).permute(
            0, 3, 1, 2
        )
        return ImageFlowPrediction(prediction, noise - clean, sigma)

    def step(self, request: Flux3FlowRequest) -> StepResult:
        """Return FP32 element-mean native flow MSE and the actual sample count."""
        result = self.predict(request)
        return StepResult(
            image_flow_loss(result.prediction, result.target, request.sigma, self.wavelet), request.latents.shape[0]
        )
