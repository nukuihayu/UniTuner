"""Bind Qwen Image target/layer/reference flow math to shared execution resources."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from unitunner.data.cache.storage import ImageCacheRecord
from unitunner.execution.regions import ModelRegion, ModelRegions
from unitunner.models.components.conditioning import Caption
from unitunner.models.qwen_image.conditioning import batch_context, pack_latents, unpack_latents
from unitunner.models.qwen_image.positions import ImageGrid
from unitunner.models.qwen_image.projections import adapter_projections
from unitunner.models.qwen_image.restoration import restore_latents
from unitunner.models.qwen_image.transformer import QwenImageTransformer
from unitunner.training.contracts import ImageFlowPrediction, StepResult
from unitunner.training.losses.wavelet import WaveletLoss, image_flow_loss


@dataclass(frozen=True, slots=True)
class QwenImageFlowRequest:
    """Borrow BCLHW clean/noise, B normalized times, LD contexts and BC1HW references.

    Layered clean values include the base image first and foregrounds afterward.
    The base is also appended as a clean condition; remove_base excludes it only
    from noisy target tokens and loss. Edit variants accept ordered references
    with independent spatial geometry, shared across the batch.
    """

    latents: Tensor
    noise: Tensor
    sigma: Tensor
    captions: tuple[Tensor, ...]
    references: tuple[Tensor, ...] = ()
    remove_base: bool = False


class QwenImageBinding:
    """Expose native block regions and active adapter targets without a second trainer."""

    def __init__(self, model: QwenImageTransformer, *, wavelet: WaveletLoss | None = None) -> None:
        self.model = model
        self.wavelet = wavelet
        blocks = tuple(ModelRegion(block, index, "visual") for index, block in enumerate(model.transformer_blocks))
        self.regions = ModelRegions(blocks)

    def adapter_targets(self) -> dict[str, nn.Linear]:
        """Borrow image-relevant projections, excluding final unused text queries/outputs."""
        return adapter_projections(self.model)

    def restore(
        self, low_latents: Tensor, captions: Tensor, *, sigma: float, caption_mask: Tensor | None = None
    ) -> Tensor:
        """Restore normalized BC1HW directly with native noise-minus-data prediction."""
        return restore_latents(self.model, low_latents, Caption(captions, caption_mask), sigma=sigma)

    def cached_step(
        self, records: list[ImageCacheRecord], noise: Tensor, sigma: Tensor, *, remove_base: bool = False
    ) -> StepResult:
        """Average per-sample native flow losses without padding independent references.

        Target CLHW geometry matches the noise batch; each sample's ordered C1HW
        references and caption length may differ. Preserve the supplied per-sample
        noise/time and every graph. With equal target shapes this equals the
        element-mean batched loss. Layered clean base is derived in predict.
        """
        if not records or noise.shape[0] != len(records) or sigma.shape != (len(records),):
            raise ValueError("Qwen cached flow requires matching nonempty records, noise and sigma batches")
        losses = []
        for index, record in enumerate(records):
            if record.latents.shape != noise.shape[1:]:
                raise ValueError("Qwen cached target geometry differs from its noise")
            losses.append(
                self.step(
                    QwenImageFlowRequest(
                        record.latents.unsqueeze(0),
                        noise[index : index + 1],
                        sigma[index : index + 1],
                        (record.caption,),
                        tuple(value.unsqueeze(0) for value in record.references),
                        remove_base,
                    )
                ).loss
            )
        return StepResult(torch.stack(losses).mean(), len(records))

    def predict(self, request: QwenImageFlowRequest) -> ImageFlowPrediction:
        """Return BCLHW velocity and noise-minus-clean target with native layered semantics.

        Interpolate unrounded normalized sigma before packing. Layered uses
        RGB condition0, one grid per target layer and the clean base last; edit
        references follow targets. Slice predictions before unpacking, so clean
        references never enter the main loss. Inputs/RNG/modes remain unchanged.
        """
        clean, noise, sigma = request.latents, request.noise, request.sigma
        if (
            clean.ndim != 5
            or min(clean.shape) < 1
            or not clean.is_floating_point()
            or any(value % 2 for value in clean.shape[-2:])
            or noise.shape != clean.shape
            or noise.device != clean.device
            or noise.dtype != clean.dtype
            or clean.shape[1] != self.model.config.out_channels
            or len(request.captions) != clean.shape[0]
        ):
            raise ValueError(
                "Qwen Image flow requires matching floating BCLHW clean/noise with native channels/even H/W"
            )
        if (
            sigma.shape != (clean.shape[0],)
            or sigma.device != clean.device
            or not sigma.is_floating_point()
            or not torch.all(torch.isfinite(sigma) & (sigma >= 0) & (sigma <= 1))
        ):
            raise ValueError("Qwen Image sigma requires a finite normalized B vector in[0,1]")
        layered = self.model.config.variant == "layered"
        if request.remove_base and not layered:
            raise ValueError("Qwen Image remove_base requires layered")
        if layered:
            if clean.shape[2] < 2 or request.references:
                raise ValueError("Qwen Image layered requires base plus foreground and derives its own clean reference")
            references = (clean[:, :, :1],)
        else:
            if clean.shape[2] != 1:
                raise ValueError("Qwen Image non-layered targets require BC1HW")
            if self.model.config.variant == "original" and request.references:
                raise ValueError("Qwen Image original flow does not consume edit references")
            references = request.references
        for reference in references:
            if (
                reference.ndim != 5
                or reference.shape[:3] != (clean.shape[0], clean.shape[1], 1)
                or min(reference.shape) < 1
                or any(value % 2 for value in reference.shape[-2:])
                or reference.dtype != clean.dtype
                or reference.device != clean.device
            ):
                raise ValueError("Qwen Image references require matching BC1HW and positive even H/W")
        fraction = sigma.to(clean.dtype)[:, None, None, None, None]
        noisy = (1 - fraction) * clean + fraction * noise
        if request.remove_base:
            clean, noise, noisy = clean[:, :, 1:], noise[:, :, 1:], noisy[:, :, 1:]
        context, mask = batch_context(request.captions)
        if context.device != clean.device or context.dtype != clean.dtype:
            raise ValueError("Qwen Image flow context must share latent dtype/device")
        tokens = pack_latents(noisy)
        target_count = tokens.shape[1]
        grids: tuple[ImageGrid, ...] = ((1, clean.shape[-2] // 2, clean.shape[-1] // 2),) * clean.shape[2]
        if references:
            tokens = torch.cat((tokens, *(pack_latents(value) for value in references)), dim=1)
            grids += tuple((1, value.shape[-2] // 2, value.shape[-1] // 2) for value in references)
        prediction = self.model(
            tokens,
            context,
            sigma,
            grids,
            text_mask=mask,
            target_count=target_count,
            layer_condition=torch.zeros(clean.shape[0], dtype=torch.int64, device=clean.device) if layered else None,
        )
        prediction = unpack_latents(
            prediction[:, :target_count], layers=clean.shape[2], height=clean.shape[-2], width=clean.shape[-1]
        )
        return ImageFlowPrediction(prediction, noise - clean, sigma)

    def step(self, request: QwenImageFlowRequest) -> StepResult:
        """Return FP32 element-mean flow MSE and the actual batch sample count."""
        result = self.predict(request)
        return StepResult(
            image_flow_loss(result.prediction, result.target, request.sigma, self.wavelet), request.latents.shape[0]
        )
