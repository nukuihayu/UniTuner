"""Bind MOSS native sigma*1000 time and noise-minus-clean flow to the shared engine."""

import torch
from torch import Tensor, nn

from unituner.execution.regions import ModelRegion, ModelRegions, RegionBranch
from unituner.models.components.conditioning import Caption
from unituner.models.moss_image.projections import adapter_projections
from unituner.models.moss_image.restoration import restore_latents
from unituner.models.moss_image.transformer import MossImageTransformer
from unituner.training.contracts import ImageFlowPrediction, ImageFlowRequest, StepResult
from unituner.training.losses.wavelet import WaveletLoss, image_flow_loss


class MossImageBinding:
    """Borrow a native transformer and expose its geometry to shared execution owners."""

    def __init__(self, model: MossImageTransformer, *, wavelet: WaveletLoss | None = None) -> None:
        self.model = model
        self.wavelet = wavelet
        branches: tuple[tuple[RegionBranch, nn.ModuleList], ...] = (
            ("condition", model.text_embedder.blocks),
            ("visual", model.layers),
        )
        blocks = tuple(
            ModelRegion(block, index, branch) for branch, blocks in branches for index, block in enumerate(blocks)
        )
        self.regions = ModelRegions(blocks)

    def adapter_targets(self) -> dict[str, nn.Linear]:
        """Borrow native caption/main attention and FFN adapter targets."""
        return adapter_projections(self.model)

    def restore(self, low_latents: Tensor, captions: Caption, *, sigma: float) -> Tensor:
        """Restore native latents with the direct mid-timestep Euler method."""
        return restore_latents(self.model, low_latents, captions, sigma=sigma)

    def predict(self, request: ImageFlowRequest) -> ImageFlowPrediction:
        """Interpolate at sigma and evaluate native velocity without detaching inputs.

        Clean/noise must have identical floating BCHW shape, dtype and device.
        Sigma retains its floating precision for native time; the interpolation
        fraction is rounded to the latent dtype. No additional time offset or
        latent normalization is applied. The caller owns randomness and modes.
        """
        latents, noise, sigma = request.latents, request.noise, request.sigma
        if (
            latents.ndim != 4
            or not latents.is_floating_point()
            or noise.shape != latents.shape
            or noise.dtype != latents.dtype
            or noise.device != latents.device
        ):
            raise ValueError("MOSS clean/noise must share floating BCHW shape, dtype and device")
        if sigma.shape != (latents.shape[0],) or sigma.device != latents.device or not sigma.is_floating_point():
            raise ValueError("MOSS flow sigma must be floating B values on the latent device")
        if not torch.all(torch.isfinite(sigma) & (sigma >= 0) & (sigma <= 1)):
            raise ValueError("MOSS flow sigma must be finite and in [0,1]")
        fraction = sigma.to(latents.dtype)[:, None, None, None]
        noisy_latents = (1 - fraction) * latents + fraction * noise
        timesteps = sigma * 1000
        prediction = self.model(
            noisy_latents, timesteps, request.captions, request.caption_mask, references=request.references
        )
        return ImageFlowPrediction(prediction, noise - latents, timesteps)

    def step(self, request: ImageFlowRequest) -> StepResult:
        """Return the FP32 element-mean velocity MSE with actual accumulation count."""
        result = self.predict(request)
        return StepResult(
            image_flow_loss(result.prediction, result.target, request.sigma, self.wavelet), request.latents.shape[0]
        )
