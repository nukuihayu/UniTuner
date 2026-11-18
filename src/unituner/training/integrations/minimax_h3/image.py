"""Native H3-Image flow matching and single-step image restoration."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unituner.execution.regions import ModelRegion, ModelRegions
from unituner.models.minimax_h3.image import MiniMaxH3ImageTransformer
from unituner.models.minimax_h3.image_inference import encode_image_condition
from unituner.models.minimax_h3.image_vae import MiniMaxH3ImageVAE
from unituner.models.minimax_h3.projections import adapter_projections
from unituner.models.minimax_h3.text.encoder import MiniMaxH3TextEncoder
from unituner.training.contracts import ImageFlowPrediction, StepResult
from unituner.training.losses.wavelet import WaveletLoss, image_flow_loss


@dataclass(frozen=True, slots=True)
class MiniMaxH3ImageFlowRequest:
    """One clean/noise B24x1xHxW sample and its pre-encoded H3 image condition."""

    latents: Tensor
    noise: Tensor
    sigma: Tensor
    text: Tensor
    token_tags: Tensor
    references: tuple[Tensor, ...] = ()
    reference_seed: int = 0


@torch.no_grad()
def prepare_image_flow_request(
    vae: MiniMaxH3ImageVAE,
    text_encoder: MiniMaxH3TextEncoder,
    pixels: Tensor,
    prompt: str,
    *,
    noise: Tensor,
    sigma: Tensor,
    model_device: torch.device,
    images: Sequence[Path] = (),
    reference_seed: int = 0,
) -> MiniMaxH3ImageFlowRequest:
    """Create a flow sample using the same reference and text encoding as image inference."""
    if pixels.ndim != 4 or pixels.shape[:2] != (1, 3) or any(size % 32 for size in pixels.shape[-2:]):
        raise ValueError("H3-Image training needs B3HW pixels on a 32-pixel canvas")
    text, tags, references, _ = encode_image_condition(
        vae,
        text_encoder,
        prompt,
        width=pixels.shape[-1],
        height=pixels.shape[-2],
        images=images,
        model_device=model_device,
    )
    latents = vae.encode(pixels.to(next(vae.parameters()).device).unsqueeze(2)).to(model_device)
    if latents.shape != noise.shape or latents.device != noise.device:
        raise ValueError("H3-Image training noise must match encoded target latents")
    return MiniMaxH3ImageFlowRequest(latents, noise, sigma, text, tags, references, reference_seed)


class MiniMaxH3ImageBinding:
    """Keep the H3 image backbone, adapter targets and training math together."""

    def __init__(self, model: MiniMaxH3ImageTransformer, *, wavelet: WaveletLoss | None = None) -> None:
        self.model = model
        self.wavelet = wavelet
        self.regions = ModelRegions(
            tuple(ModelRegion(block, index, "visual") for index, block in enumerate(model.blocks))
        )

    def adapter_targets(self) -> dict[str, nn.Linear]:
        return adapter_projections(self.model)

    def _predict(
        self,
        noisy: Tensor,
        text: Tensor,
        tags: Tensor,
        sigma: float,
        references: tuple[Tensor, ...],
        reference_seed: int,
    ) -> Tensor:
        next_sigma = next((value for value in self.model.pdd_sigmas[1:] if value < sigma), 0.0)
        return self.model.predict(
            noisy,
            text,
            tags,
            sigma=sigma,
            sigma_next=next_sigma,
            references=references,
            reference_seed=reference_seed,
        )

    def predict(self, request: MiniMaxH3ImageFlowRequest) -> ImageFlowPrediction:
        """Match dataward velocity on a continuous noise fraction using the nearest PDD interval."""
        clean, noise, sigma = request.latents, request.noise, request.sigma
        if (
            clean.ndim != 5
            or clean.shape[:3] != (1, 24, 1)
            or clean.shape != noise.shape
            or clean.dtype != noise.dtype
            or clean.device != noise.device
            or sigma.shape != (1,)
            or sigma.device != clean.device
            or not sigma.is_floating_point()
            or not torch.all(torch.isfinite(sigma) & (sigma > 0) & (sigma <= 1))
        ):
            raise ValueError("H3-Image flow needs matching B24x1xHxW latents and a positive noise fraction")
        fraction = sigma.to(clean.dtype).reshape(1, 1, 1, 1, 1)
        noisy = (1 - fraction) * clean + fraction * noise
        velocity = self._predict(
            noisy, request.text, request.token_tags, float(sigma.item()), request.references, request.reference_seed
        )
        return ImageFlowPrediction(velocity, clean - noise, sigma)

    def step(self, request: MiniMaxH3ImageFlowRequest) -> StepResult:
        result = self.predict(request)
        return StepResult(image_flow_loss(result.prediction, result.target, request.sigma, self.wavelet), 1)

    def restore(
        self,
        low_latents: Tensor,
        text: Tensor,
        token_tags: Tensor,
        *,
        sigma: float,
        references: tuple[Tensor, ...] = (),
        reference_seed: int = 0,
    ) -> Tensor:
        """Predict a clean normalized latent directly from an LQ latent at sigma."""
        if not 0 < sigma <= 1 or low_latents.shape[:3] != (1, 24, 1):
            raise ValueError("H3-Image restoration needs a single B24x1xHxW input and 0<sigma<=1")
        return (
            low_latents.float()
            + sigma
            * self._predict(
                low_latents,
                text,
                token_tags,
                sigma,
                references,
                reference_seed,
            ).float()
        )

    def restoration_step(
        self,
        low_latents: Tensor,
        high_latents: Tensor,
        text: Tensor,
        token_tags: Tensor,
        *,
        sigma: float,
        references: tuple[Tensor, ...] = (),
        reference_seed: int = 0,
    ) -> StepResult:
        """Supervise a one-step normalized-latent restoration without detaching the encoder graph."""
        if high_latents.shape != low_latents.shape or high_latents.device != low_latents.device:
            raise ValueError("H3-Image restoration targets must match low latents")
        restored = self.restore(
            low_latents, text, token_tags, sigma=sigma, references=references, reference_seed=reference_seed
        )
        return StepResult(F.mse_loss(restored.float(), high_latents.float()), 1)

    def restoration_image_step(
        self,
        vae: MiniMaxH3ImageVAE,
        low: Tensor,
        high: Tensor,
        text: Tensor,
        token_tags: Tensor,
        *,
        sigma: float,
        references: tuple[Tensor, ...] = (),
        reference_seed: int = 0,
    ) -> StepResult:
        """Train one-step restoration through the image VAE with an RGB reconstruction loss."""
        if low.shape != high.shape or low.ndim != 4 or low.shape[:2] != (1, 3):
            raise ValueError("H3-Image restoration requires matching B3HW low/high images")
        low_latents = vae.encode(low.unsqueeze(2))
        generated = vae.decode(
            self.restore(
                low_latents, text, token_tags, sigma=sigma, references=references, reference_seed=reference_seed
            )
        )[:, :, 0]
        return StepResult(F.mse_loss(generated.float(), high.float()), 1, "generator")
