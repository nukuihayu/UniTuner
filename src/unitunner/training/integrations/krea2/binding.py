"""Bind native Krea2 flow mathematics to shared execution regions and adapter ownership."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from unitunner.execution.regions import ModelRegion, ModelRegions
from unitunner.models.krea2.conditioning import (
    Krea2TextFeatures,
    batch_text,
    image_positions,
    pack_latents,
    unpack_latents,
)
from unitunner.models.krea2.projections import adapter_projections
from unitunner.models.krea2.restoration import restore_latents
from unitunner.models.krea2.transformer import Krea2Transformer
from unitunner.training.contracts import ImageFlowPrediction, StepResult
from unitunner.training.losses.wavelet import WaveletLoss, image_flow_loss


@dataclass(frozen=True, slots=True)
class Krea2FlowRequest:
    """Borrow BC1HW normalized latents/noise, B sigma and compact SLD caption rows.

    timesteps optionally carries native0..1000 scheduler values, independently
    of interpolation sigma. Default sigma*1000 preserves actual trainer order.
    """

    latents: Tensor
    noise: Tensor
    sigma: Tensor
    captions: tuple[Tensor, ...]
    timesteps: Tensor | None = None


class Krea2Binding:
    """Expose visual, selected-layer text and text-refiner regions without a second trainer."""

    def __init__(self, model: Krea2Transformer, *, wavelet: WaveletLoss | None = None) -> None:
        self.model = model
        self.wavelet = wavelet
        blocks = tuple(
            [ModelRegion(block, index, "text") for index, block in enumerate(model.txtfusion.layerwise_blocks)]
            + [ModelRegion(block, index, "refiner") for index, block in enumerate(model.txtfusion.refiner_blocks)]
            + [ModelRegion(block, index, "visual") for index, block in enumerate(model.blocks)]
        )
        self.regions = ModelRegions(blocks)

    def adapter_targets(self) -> dict[str, nn.Linear]:
        """Borrow all native linears, including text/time/final projections."""
        return adapter_projections(self.model)

    def restore(self, low_latents: Tensor, captions: Krea2TextFeatures, *, sigma: float) -> Tensor:
        """Restore BC1HW directly with selected-layer captions and native scheduler time."""
        return restore_latents(self.model, low_latents, captions, sigma=sigma)

    def predict(self, request: Krea2FlowRequest) -> ImageFlowPrediction:
        """Return BC1HW velocity/target and native0..1000 timesteps.

        Interpolate clean*(1-sigma)+noise*sigma, flatten C/patch-row/patch-column,
        and keep image tokens first. Text is compacted by the cache/text caller;
        host prefix lengths exclude padding from attention. Preserve all active
        input/model gradients, native noise-minus-data sign and RNG ownership.
        """
        clean, noise, sigma = request.latents, request.noise, request.sigma
        if (
            clean.ndim != 5
            or min(clean.shape) < 1
            or clean.shape[1:3] != (self.model.config.in_channels, 1)
            or not clean.is_floating_point()
            or clean.shape != noise.shape
            or noise.dtype != clean.dtype
            or noise.device != clean.device
            or len(request.captions) != clean.shape[0]
            or sigma.shape != (clean.shape[0],)
            or not sigma.is_floating_point()
            or sigma.device != clean.device
            or not torch.all(torch.isfinite(sigma) & (sigma >= 0) & (sigma <= 1))
        ):
            raise ValueError(
                "Krea2 flow requires matching floating BC1HW clean/noise, B sigma and one SLD caption per sample"
            )
        text = batch_text(request.captions)
        fraction = sigma.to(clean.dtype)[:, None, None, None, None]
        noisy = clean * (1 - fraction) + noise * fraction
        timesteps = sigma * 1000 if request.timesteps is None else request.timesteps
        if request.timesteps is not None and (
            timesteps.shape != sigma.shape
            or timesteps.device != clean.device
            or not timesteps.is_floating_point()
            or not torch.all(torch.isfinite(timesteps) & (timesteps >= 0) & (timesteps <= 1000))
        ):
            raise ValueError("Krea2 timesteps require finite native0..1000 B values on the latent device")
        pixels = noisy[:, :, 0]
        patch = self.model.config.patch_size
        prediction = self.model(
            pack_latents(pixels, patch=patch),
            text.features,
            timesteps / 1000,
            image_positions(pixels, patch=patch, text_count=text.features.shape[1]),
            text_lengths=text.lengths,
        )
        return ImageFlowPrediction(
            unpack_latents(prediction, patch=patch, height=pixels.shape[-2], width=pixels.shape[-1])[:, :, None],
            noise - clean,
            timesteps,
        )

    def step(self, request: Krea2FlowRequest) -> StepResult:
        """Return FP32 element-mean flow MSE and actual microbatch sample count."""
        result = self.predict(request)
        return StepResult(
            image_flow_loss(result.prediction, result.target, request.sigma, self.wavelet), request.latents.shape[0]
        )
