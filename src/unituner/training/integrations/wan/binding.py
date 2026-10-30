"""Bind native Wan video noising, explicit model time and conditions to the shared engine."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from unituner.execution.regions import ModelRegion, ModelRegions
from unituner.models.wan.data.cache.storage import WanCacheRecord
from unituner.models.wan.projections import adapter_projections
from unituner.models.wan.transformer import WanTransformer
from unituner.training.contracts import StepResult


@dataclass(frozen=True, slots=True)
class WanFlowRequest:
    """Carry BCFHW clean/noise, B sigma/native times, per-row LD text and conditions.

    Sigma is a noise fraction in[0,1]. Timesteps are independent native scheduler
    units: the original direct sampler computes sigma*1000 then adds1, whereas
    a discrete scheduler supplies its own times. Never reconstruct sigma by
    dividing the model time. Captions stay unpadded until the native transformer.
    Image conditions are B20FHW; CLIP is B2571280 or explicit FLF2V B22571280.
    Optional control latents are B16FHW. Explicit host coordinates describe
    independent-frame rotary slots; they do not request a target-only loss.
    """

    latents: Tensor
    noise: Tensor
    sigma: Tensor
    timesteps: Tensor
    captions: tuple[Tensor, ...]
    image_condition: Tensor | None = None
    image_features: Tensor | None = None
    control_latents: Tensor | None = None
    frame_indices: tuple[tuple[int, ...], ...] | None = None


@dataclass(frozen=True, slots=True)
class WanFlowPrediction:
    """Return native BCFHW velocity/target, B times and assembled differentiable inputs."""

    prediction: Tensor
    target: Tensor
    timesteps: Tensor
    noisy_latents: Tensor
    conditioning: Tensor | None


class WanBinding:
    """Borrow a native Wan expert and expose block regions without owning sampling.

    This component performs deterministic math only. Expert selection, noise/time
    sampling, cache residency, adapter publication and resume belong to callers.
    No RCM mask or target-slot-only loss is constructed or consumed.
    """

    def __init__(self, model: WanTransformer) -> None:
        self.model = model
        blocks = tuple(ModelRegion(block, index, "visual") for index, block in enumerate(model.blocks))
        self.regions = ModelRegions(blocks)

    def adapter_targets(self) -> dict[str, nn.Linear]:
        """Borrow native block Linear targets without embeddings, head or modulation."""
        return adapter_projections(self.model)

    def cached_step(self, records: list[WanCacheRecord], noise: Tensor, sigma: Tensor, timesteps: Tensor) -> StepResult:
        """Read an equal-grid batch without mutating cache tensors or drawing RNG.

        Independent-frame single-image features retain an explicit clean axis
        on disk; ordinary I2V removes that unit axis only. Two-image features
        require the explicit FLF2V topology. Coordinates must be present for
        every row or absent for every row. Storage owners validate records.
        """
        if not records:
            raise ValueError("Wan cached training requires a nonempty batch")
        images, features, controls, coordinates = [], [], [], []
        for record in records:
            if record.image_condition is not None:
                images.append(record.image_condition)
            if record.image_features is not None:
                feature = record.image_features
                if self.model.config.variant == "i2v" and feature.ndim == 3 and feature.shape[0] == 1:
                    feature = feature[0]
                features.append(feature)
            if record.control_latents is not None:
                controls.append(record.control_latents)
            if record.frame_indices is not None:
                coordinates.append(record.frame_indices)
        if any(values and len(values) != len(records) for values in (images, features, controls, coordinates)):
            raise ValueError("Wan cached training requires a uniform condition/coordinate inventory")
        return self.step(
            WanFlowRequest(
                torch.stack([record.latents for record in records]),
                noise,
                sigma,
                timesteps,
                tuple(record.caption for record in records),
                None if not images else torch.stack(images),
                None if not features else torch.stack(features),
                None if not controls else torch.stack(controls),
                None if not coordinates else tuple(coordinates),
            )
        )

    def predict(self, request: WanFlowRequest) -> WanFlowPrediction:
        """Predict noise-minus-clean at independently supplied native model times.

        Interpolate (1-sigma)*clean + sigma*noise in FP32. Native I2V conditions
        concatenate before patch embedding. Fun-Control concatenates control
        latents and a first-frame-only image mean (or zeros for T2V), stripping
        four visibility channels without modifying any input/view. Loss targets
        include all temporal slots, matching the actual original flow caller.
        Preserve gradients to all active inputs; caller owns frozen/train modes.
        """
        clean, noise, sigma = request.latents, request.noise, request.sigma
        if (
            clean.ndim != 5
            or min(clean.shape) < 1
            or clean.shape[1] != self.model.config.out_channels
            or clean.dtype != torch.float32
            or noise.shape != clean.shape
            or noise.device != clean.device
            or noise.dtype != clean.dtype
            or len(request.captions) != clean.shape[0]
        ):
            raise ValueError("Wan flow requires matching FP32 BCFHW clean/noise and per-row captions")
        for name, value, maximum in (("sigma", sigma, 1), ("timesteps", request.timesteps, 1001)):
            if (
                value.shape != (clean.shape[0],)
                or value.device != clean.device
                or value.dtype != torch.float32
                or not torch.all(torch.isfinite(value) & (value >= 0) & (value <= maximum))
            ):
                raise ValueError(f"Wan {name} requires finite FP32 B values in[0,{maximum}] on the latent device")
        image = request.image_condition
        requires_image = self.model.config.variant in ("i2v", "flf2v")
        if (image is not None) != requires_image:
            raise ValueError("Wan image condition must match the explicit I2V/FLF2V topology")
        if image is not None and (
            image.shape != (clean.shape[0], 20, *clean.shape[2:])
            or image.dtype != clean.dtype
            or image.device != clean.device
        ):
            raise ValueError("Wan image condition requires matching FP32 B20FHW")
        condition = image
        if request.control_latents is not None:
            control = request.control_latents
            if control.shape != clean.shape or control.dtype != clean.dtype or control.device != clean.device:
                raise ValueError("Wan control latents require matching FP32 BCFHW")
            first = (
                torch.zeros_like(control)
                if image is None
                else torch.cat((image[:, 4:, :1], torch.zeros_like(image[:, 4:, 1:])), dim=2)
            )
            condition = torch.cat((control, first), dim=1)
        fraction = sigma[:, None, None, None, None]
        noisy = (1 - fraction) * clean + fraction * noise
        sequence_length = math.prod(clean.shape[2:]) // math.prod(self.model.config.patch_size)
        prediction = self.model(
            tuple(noisy.unbind()),
            request.timesteps,
            request.captions,
            sequence_length=sequence_length,
            conditioning=None if condition is None else tuple(condition.unbind()),
            image_features=request.image_features,
            frame_indices=request.frame_indices,
        )
        return WanFlowPrediction(torch.stack(prediction), noise - clean, request.timesteps, noisy, condition)

    def step(self, request: WanFlowRequest) -> StepResult:
        """Return FP32 element-mean native flow MSE and actual batch sample count."""
        result = self.predict(request)
        return StepResult(F.mse_loss(result.prediction.float(), result.target.float()), request.latents.shape[0])
