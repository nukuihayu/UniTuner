"""Bind Z-Image reverse time and velocity to a shared flow-matching update."""

import torch
from torch import Tensor, nn

from unituner.execution.regions import ModelRegion, ModelRegions, RegionBranch
from unituner.models.components.conditioning import Caption
from unituner.models.z_image.projections import adapter_projections
from unituner.models.z_image.restoration import restore_latents
from unituner.models.z_image.transformer import ZImageTransformer
from unituner.training.contracts import ImageFlowPrediction, ImageFlowRequest, StepResult
from unituner.training.losses.wavelet import WaveletLoss, image_flow_loss


class ZImageBinding:
    """Bind one borrowed model to explicit geometry and the local training engine."""

    def __init__(self, model: ZImageTransformer, *, wavelet: WaveletLoss | None = None) -> None:
        self.model = model
        self.wavelet = wavelet
        branches: tuple[tuple[RegionBranch, nn.ModuleList], ...] = (
            ("refiner", model.noise_refiner),
            ("condition", model.context_refiner),
            ("visual", model.layers),
        )
        blocks = tuple(
            ModelRegion(block, index, branch) for branch, blocks in branches for index, block in enumerate(blocks)
        )
        self.regions = ModelRegions(blocks)

    def adapter_targets(self) -> dict[str, nn.Linear]:
        """Select main-block attention/FFN projections, excluding refiners/modulation.

        Names are native artifact paths. The explicit default inventory matches
        the pinned Musubi Z-Image target policy; selection alone does not prove
        adapter training or external-consumer parity.
        """
        return adapter_projections(self.model)

    def predict(self, request: ImageFlowRequest) -> ImageFlowPrediction:
        """Interpolate at sigma and predict with native time 1-sigma, without detaching."""
        if request.references:
            raise ValueError("Z-Image does not support reference conditioning")
        latents, noise, sigma = request.latents, request.noise, request.sigma
        if (
            latents.ndim != 4
            or noise.shape != latents.shape
            or noise.dtype != latents.dtype
            or noise.device != latents.device
        ):
            raise ValueError("Z-Image clean/noise must share floating [B,C,H,W] shape, dtype and device")
        if sigma.shape != (latents.shape[0],) or sigma.device != latents.device or not sigma.is_floating_point():
            raise ValueError("Flow sigma must be floating [B] on the latent device")
        if not torch.all(torch.isfinite(sigma) & (sigma >= 0) & (sigma <= 1)):
            raise ValueError("Flow sigma must be finite and in [0,1]")
        fraction = sigma.to(latents.dtype)[:, None, None, None]
        noisy_latents = (1 - fraction) * latents + fraction * noise
        timesteps = 1 - sigma
        prediction = self.model(noisy_latents.unsqueeze(2), timesteps, request.captions, request.caption_mask).squeeze(
            2
        )
        return ImageFlowPrediction(prediction, latents - noise, timesteps)

    def step(self, request: ImageFlowRequest) -> StepResult:
        """Return the FP32 element-mean native velocity MSE for sample accumulation."""
        result = self.predict(request)
        loss = image_flow_loss(result.prediction, result.target, request.sigma, self.wavelet)
        return StepResult(loss, request.latents.shape[0])

    def restore(
        self, low_latents: Tensor, captions: Tensor, *, sigma: float, caption_mask: Tensor | None = None
    ) -> Tensor:
        """Restore normalized LQ BCHW latents with one native data-minus-noise step.

        Feed the LQ encoding directly to the model at native reverse time 1-sigma;
        do not inject a second noise tensor. Return FP32 latents, preserving the
        graph to encoder adapters and denoiser parameters. The caller owns VAE
        decoding and image/latent losses. Sigma is strictly inside (0,1).
        """
        return restore_latents(self.model, low_latents, Caption(captions, caption_mask), sigma=sigma)
