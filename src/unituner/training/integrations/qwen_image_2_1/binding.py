"""Bind native Qwen2.1 flow and ordered image slots to the shared update engine."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from unituner.execution.regions import ModelRegion, ModelRegions
from unituner.models.components.conditioning import Caption, batch_captions
from unituner.models.qwen_image_2_1.conditioning import pack_latents, unpack_latents
from unituner.models.qwen_image_2_1.data.cache.storage import QwenImage21CacheRecord
from unituner.models.qwen_image_2_1.projections import adapter_projections
from unituner.models.qwen_image_2_1.restoration import restore_latents
from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer
from unituner.training.contracts import ImageFlowPrediction, StepResult
from unituner.training.losses.wavelet import WaveletLoss, image_flow_loss


@dataclass(frozen=True, slots=True)
class QwenImage21FlowRequest:
    """Borrow BC1HW clean/noise, B sigma, LD captions, BC1HW references and host slots.

    Reference geometry may differ across ordered references; one batched request
    shares each reference's H/W. Cached samples with independent geometries are
    evaluated separately while retaining their noise/time and mean loss weight.
    """

    latents: Tensor
    noise: Tensor
    sigma: Tensor
    captions: tuple[Tensor, ...]
    references: tuple[Tensor, ...] = ()
    image_slots: tuple[tuple[int, ...], ...] = ()
    timesteps: Tensor | None = None


class QwenImage21Binding:
    """Expose all native blocks and seven active adapter projections per block."""

    def __init__(self, model: QwenImage21Transformer, *, wavelet: WaveletLoss | None = None) -> None:
        self.model = model
        self.wavelet = wavelet
        blocks = tuple(ModelRegion(block, index, "visual") for index, block in enumerate(model.transformer_blocks))
        self.regions = ModelRegions(blocks)

    def adapter_targets(self) -> dict[str, nn.Linear]:
        """Borrow native attention/MLP projections without installing training policy."""
        return adapter_projections(self.model)

    def restore(self, low_latents: Tensor, captions: Caption, *, sigma: float) -> Tensor:
        """Restore native BC1HW directly, borrowing host ragged-caption lengths."""
        return restore_latents(self.model, low_latents, captions, sigma=sigma)

    def predict(self, request: QwenImage21FlowRequest) -> ImageFlowPrediction:
        """Return BC1HW native velocity/target and B native0..1000 timesteps.

        Interpolate (1-sigma)*clean+sigma*noise before row-major token packing.
        Text remains unpadded until batching; references are inserted at each
        row's explicit text offsets, with their own grids before target grids.
        Only target noise-minus-clean participates in loss. Input/reference/text
        and parameter gradients are retained; no mode/RNG mutations occur.
        Explicit timesteps use the trainer's native0..1000 unit, independently
        of sigma's interpolation fraction. Default timesteps are sigma*1000;
        divide by1000 at the model boundary, preserving native CUDA rounding.
        """
        clean, noise, sigma = request.latents, request.noise, request.sigma
        if (
            clean.ndim != 5
            or min(clean.shape) < 1
            or clean.shape[2] != 1
            or not clean.is_floating_point()
            or clean.shape[1] != self.model.config.in_channels
            or self.model.config.out_channels != clean.shape[1]
            or noise.shape != clean.shape
            or noise.dtype != clean.dtype
            or noise.device != clean.device
            or len(request.captions) != clean.shape[0]
        ):
            raise ValueError(
                "Qwen Image2.1 flow requires matching floating BC1HW clean/noise and one LD caption per sample"
            )
        if (
            sigma.shape != (clean.shape[0],)
            or sigma.device != clean.device
            or not sigma.is_floating_point()
            or not torch.all(torch.isfinite(sigma) & (sigma >= 0) & (sigma <= 1))
        ):
            raise ValueError("Qwen Image2.1 sigma requires finite normalized B values in[0,1]")
        for reference in request.references:
            if (
                reference.ndim != 5
                or reference.shape[:3] != clean.shape[:3]
                or min(reference.shape) < 1
                or reference.device != clean.device
                or reference.dtype != clean.dtype
            ):
                raise ValueError(
                    "Qwen Image2.1 references require matching floating BC1HW with independent positive H/W"
                )
        context = batch_captions(request.captions).features
        if context.device != clean.device or context.dtype != clean.dtype:
            raise ValueError("Qwen Image2.1 captions must share latent dtype/device")
        fraction = sigma.to(clean.dtype)[:, None, None, None, None]
        noisy = (1 - fraction) * clean + fraction * noise
        timesteps = sigma * 1000 if request.timesteps is None else request.timesteps
        # Derived times inherit the already validated sigma range. Only caller-
        # supplied native times need a second device reduction/readback.
        if request.timesteps is not None and (
            timesteps.shape != sigma.shape
            or timesteps.device != sigma.device
            or not timesteps.is_floating_point()
            or not torch.all(torch.isfinite(timesteps) & (timesteps >= 0) & (timesteps <= 1000))
        ):
            raise ValueError("Qwen Image2.1 timesteps require finite native0..1000 B values on the latent device")
        grids = (
            *((1, reference.shape[-2], reference.shape[-1]) for reference in request.references),
            (1, clean.shape[-2], clean.shape[-1]),
        )
        slots = request.image_slots or ((),) * clean.shape[0]
        prediction = self.model(
            pack_latents(noisy),
            context,
            timesteps / 1000,
            (grids,),
            text_lengths=tuple(row.shape[0] for row in request.captions),
            references=tuple(pack_latents(value) for value in request.references),
            image_slots=slots,
        )
        return ImageFlowPrediction(
            unpack_latents(prediction, height=clean.shape[-2], width=clean.shape[-1]), noise - clean, timesteps
        )

    def step(self, request: QwenImage21FlowRequest) -> StepResult:
        """Return FP32 element-mean native flow MSE and the actual batch sample count."""
        result = self.predict(request)
        return StepResult(
            image_flow_loss(result.prediction, result.target, request.sigma, self.wavelet), request.latents.shape[0]
        )

    def cached_step(self, records: list[QwenImage21CacheRecord], noise: Tensor, sigma: Tensor) -> StepResult:
        """Average per-sample flow losses, retaining independent reference geometries.

        Records were validated during manifest/read preflight. Slot metadata
        remains on the host; no device-to-host scalar transfers occur per token.
        """
        if not records or noise.shape[0] != len(records) or sigma.shape != (len(records),):
            raise ValueError("Qwen Image2.1 cached flow requires matching nonempty records/noise/time batches")
        losses = []
        for index, record in enumerate(records):
            if record.latents.shape != noise.shape[1:] or record.image_slots.device.type != "cpu":
                raise ValueError(
                    "Qwen Image2.1 cached geometry must match noise and insertion slots must remain on CPU"
                )
            losses.append(
                self.step(
                    QwenImage21FlowRequest(
                        record.latents[None],
                        noise[index : index + 1],
                        sigma[index : index + 1],
                        (record.caption,),
                        tuple(value[None] for value in record.references),
                        (tuple(record.image_slots.tolist()),),
                    )
                ).loss
            )
        return StepResult(torch.stack(losses).mean(), len(records))
