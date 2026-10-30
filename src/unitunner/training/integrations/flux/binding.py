"""Bind native FLUX packing, pooled text and variant conditions to the shared engine."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from unitunner.data.cache.storage import ImageCacheRecord
from unitunner.execution.regions import ModelRegion, ModelRegions
from unitunner.models.flux.conditioning import (
    FluxCondition,
    FluxTextFeatures,
    build_inputs,
    image_ids,
    pack_latents,
    unpack_latents,
)
from unitunner.models.flux.projections import adapter_projections
from unitunner.models.flux.restoration import restore_latents
from unitunner.models.flux.transformer import FluxTransformer
from unitunner.training.contracts import ImageFlowPrediction, StepResult
from unitunner.training.losses.wavelet import WaveletLoss, image_flow_loss


@dataclass(frozen=True, slots=True)
class FluxFlowRequest:
    """Carry BCHW clean/noise, B sigmas, BLD T5, BV pooled CLIP and B guidance.

    Latents already use native VAE normalization. Optional conditions contain
    explicitly packed Kontext references or Fill masked-image/mask channels.
    """

    latents: Tensor
    noise: Tensor
    sigma: Tensor
    captions: Tensor
    pooled: Tensor
    guidance: Tensor | None
    condition: FluxCondition | None = None


class FluxBinding:
    """Borrow native FLUX weights and expose double/single logical execution regions.

    Parameter ownership remains with the shared full/adapter installation.
    """

    def __init__(self, model: FluxTransformer, *, wavelet: WaveletLoss | None = None) -> None:
        self.model = model
        self.wavelet = wavelet
        blocks = tuple(
            ModelRegion(block, index, "visual")
            for index, block in enumerate((*model.double_blocks, *model.single_blocks))
        )
        self.regions = ModelRegions(blocks)

    def adapter_targets(self) -> dict[str, nn.Linear]:
        """Borrow native fused attention/FFN targets without modulation parameters."""
        return adapter_projections(self.model)

    def cached_step(
        self, records: list[ImageCacheRecord], noise: Tensor, sigma: Tensor, *, guidance: float
    ) -> StepResult:
        """Assemble native cached text/conditions and reuse the same flow step.

        T5 padding stays active. Kontext references are padded only after native
        packing, with explicit valid lengths and IDs; Fill mask packing retains
        native channel order. No randomness or cache tensor mutation occurs.
        """
        latents = torch.stack([record.latents for record in records])
        captions = torch.stack([record.caption for record in records])
        pooled = []
        controls, positions, masks = [], [], []
        for record in records:
            if record.pooled is None:
                raise ValueError("Flux cached training requires CLIP pooled features")
            pooled.append(record.pooled)
            if self.model.config.variant != "dev":
                control = record.control_latents
                if control is None:
                    raise ValueError("Flux Kontext/Fill cache is missing condition latents")
                controls.append(pack_latents(control.unsqueeze(0))[0])
                positions.append(
                    image_ids(1, control.shape[1] // 2, control.shape[2] // 2, device=control.device, reference=True)[0]
                )
            if self.model.config.variant == "fill":
                if record.fill_mask is None:
                    raise ValueError("Flux Fill cache is missing mask features")
                masks.append(pack_latents(record.fill_mask.unsqueeze(0))[0])
        condition = None
        if self.model.config.variant == "kontext":
            condition = FluxCondition(
                pad_sequence(controls, batch_first=True),
                pad_sequence(positions, batch_first=True),
                tuple(value.shape[0] for value in controls),
            )
        elif self.model.config.variant == "fill":
            condition = FluxCondition(torch.stack(controls), fill_mask=torch.stack(masks))
        vector = (
            torch.full((latents.shape[0],), guidance, device=latents.device, dtype=latents.dtype)
            if self.model.config.guidance_embed
            else None
        )
        return self.step(FluxFlowRequest(latents, noise, sigma, captions, torch.stack(pooled), vector, condition))

    def restore(self, low_latents: Tensor, captions: FluxTextFeatures, *, sigma: float) -> Tensor:
        """Restore native latents with the direct mid-timestep Euler method."""
        return restore_latents(self.model, low_latents, captions, sigma=sigma)

    def predict(self, request: FluxFlowRequest) -> ImageFlowPrediction:
        """Return BCHW noise-minus-clean flow at native sigma time.

        Match the original interpolation, pack channel-row-column 2x2 cells,
        append explicit variant conditions and discard reference predictions.
        Inputs and RNG are caller-owned; no automatic conditioning or normalization
        is invented. Sigma is finite in [0,1], with no scheduler time offset.
        """
        clean, noise, sigma = request.latents, request.noise, request.sigma
        if clean.ndim != 4 or noise.shape != clean.shape or noise.device != clean.device or noise.dtype != clean.dtype:
            raise ValueError("FLUX clean/noise must share BCHW shape, dtype and device")
        if (
            sigma.shape != (clean.shape[0],)
            or sigma.device != clean.device
            or not sigma.is_floating_point()
            or not torch.all(torch.isfinite(sigma) & (sigma >= 0) & (sigma <= 1))
        ):
            raise ValueError("FLUX sigma must be floating finite B values in [0,1] on the latent device")
        fraction = sigma.to(clean.dtype)[:, None, None, None]
        noisy = pack_latents((1 - fraction) * clean + fraction * noise)
        ids = image_ids(clean.shape[0], clean.shape[2] // 2, clean.shape[3] // 2, device=clean.device)
        inputs = build_inputs(self.model.config.variant, noisy, ids, request.condition)
        txt_ids = torch.zeros((*request.captions.shape[:2], 3), device=clean.device)
        prediction = self.model(
            inputs.tokens,
            inputs.ids,
            request.captions,
            txt_ids,
            sigma,
            request.pooled,
            request.guidance,
            control_lengths=inputs.control_lengths,
        )
        prediction = unpack_latents(prediction[:, : noisy.shape[1]], clean.shape[2], clean.shape[3])
        return ImageFlowPrediction(prediction, noise - clean, sigma)

    def step(self, request: FluxFlowRequest) -> StepResult:
        """Return FP32 element-mean native velocity MSE with actual sample count."""
        result = self.predict(request)
        return StepResult(
            image_flow_loss(result.prediction, result.target, request.sigma, self.wavelet), request.latents.shape[0]
        )
