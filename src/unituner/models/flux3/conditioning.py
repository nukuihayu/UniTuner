# Derived from Black Forest Labs FLUX3 Image Klein, Apache-2.0.
"""Pack three native FLUX3 modes with explicit sample bounds and unrounded time rows."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from unituner.models.flux3.vae import image_ids as vae_image_ids

type Flux3Mode = Literal["image", "image_cond", "text"]
type Flux3Bounds = tuple[tuple[int, int], ...]
MODES: tuple[Flux3Mode, ...] = ("image", "image_cond", "text")


@dataclass(frozen=True, slots=True)
class Flux3Tokens:
    """Carry differentiable unbatched LD tokens and L4 native positions."""

    values: Tensor
    ids: Tensor


@dataclass(frozen=True, slots=True)
class Flux3Sample:
    """Describe one target, its text (including encoder padding) and optional references."""

    image: Flux3Tokens
    text: Flux3Tokens
    references: Flux3Tokens | None = None


@dataclass(frozen=True, slots=True)
class Flux3Packed:
    """Carry mode-major LD data, L4 IDs, L time rows and host sample bounds.

    Every bound is a contiguous half-open slice. Empty references have zero
    length; the transformer owns its private phantom token. Target and text
    sequences are nonempty for every sample. Callers retain tensor ownership.
    """

    data: Mapping[Flux3Mode, Tensor]
    ids: Mapping[Flux3Mode, Tensor]
    timesteps: Mapping[Flux3Mode, Tensor]
    bounds: Mapping[Flux3Mode, Flux3Bounds]

    @property
    def num_samples(self) -> int:
        """Return the number of independently attending target samples."""
        return len(self.bounds["image"])


def image_tokens(latents: Tensor, *, reference: int | None = None) -> Flux3Tokens:
    """View normalized CHW as LD with (0,row/2,column/2,reference ordinal) IDs.

    Ordered references use last-axis ordinals -1 through -10. This scaling is a
    transformer convention, distinct from the VAE's integral decoding IDs.
    No latent scaling, noise, detaching or dtype conversion is performed.
    """
    if latents.ndim != 3 or min(latents.shape) < 1 or not latents.is_floating_point():
        raise ValueError("FLUX3 image tokens require nonempty floating CHW latents")
    if reference is not None and (type(reference) is not int or not 0 <= reference < 10):
        raise ValueError("FLUX3 reference ordinal must be an integer from zero to nine")
    ids = vae_image_ids(*latents.shape[-2:], device=latents.device).float()
    ids[:, 1:3] *= 0.5
    if reference is not None:
        ids[:, 3] = -(reference + 1)
    return Flux3Tokens(latents.flatten(1).transpose(0, 1), ids)


def text_tokens(features: Tensor) -> Flux3Tokens:
    """Associate nonempty LD text features with (0,0,0,token position) IDs."""
    if features.ndim != 2 or min(features.shape) < 1 or not features.is_floating_point():
        raise ValueError("FLUX3 text tokens require nonempty floating LD features")
    positions = torch.arange(features.shape[0], device=features.device)
    zeros = torch.zeros_like(positions)
    return Flux3Tokens(features, torch.stack((zeros, zeros, zeros, positions), dim=-1))


def reference_tokens(latents: Sequence[Tensor]) -> Flux3Tokens | None:
    """Concatenate up to ten reference images without mixing reference ordinal axes."""
    if len(latents) > 10:
        raise ValueError("FLUX3 supports at most ten reference images")
    if not latents:
        return None
    parts = [image_tokens(value, reference=index) for index, value in enumerate(latents)]
    return Flux3Tokens(torch.cat([part.values for part in parts]), torch.cat([part.ids for part in parts]))


def pack_samples(samples: Sequence[Flux3Sample], timesteps: Tensor) -> Flux3Packed:
    """Concatenate three modes and isolate each sample's actual token lengths.

    Only targets receive supplied B noise fractions; text/reference time is zero.
    Preserve the supplied floating time dtype without hidden BF16 rounding. No
    batch padding is added or encoder-added text padding removed. All gradient
    paths survive concatenation and time expansion; global RNG is untouched.
    """
    if not samples:
        raise ValueError("FLUX3 packing requires at least one sample")
    device = samples[0].image.values.device
    if timesteps.shape != (len(samples),) or not timesteps.is_floating_point() or timesteps.device != device:
        raise ValueError("FLUX3 timesteps must be a floating B vector on the token device")
    widths = {
        "image": samples[0].image.values.shape[-1],
        "image_cond": samples[0].image.values.shape[-1],
        "text": samples[0].text.values.shape[-1],
    }
    values: dict[Flux3Mode, list[Tensor]] = {mode: [] for mode in MODES}
    positions: dict[Flux3Mode, list[Tensor]] = {mode: [] for mode in MODES}
    times: dict[Flux3Mode, list[Tensor]] = {mode: [] for mode in MODES}
    bounds: dict[Flux3Mode, list[tuple[int, int]]] = {mode: [] for mode in MODES}
    counts: dict[Flux3Mode, int] = dict.fromkeys(MODES, 0)
    dtype = samples[0].image.values.dtype
    for index, sample in enumerate(samples):
        tokens_by_mode = {"image": sample.image, "image_cond": sample.references, "text": sample.text}
        for mode in MODES:
            part = tokens_by_mode[mode]
            start = counts[mode]
            if part is not None:
                if (
                    part.values.ndim != 2
                    or part.values.shape[1] != widths[mode]
                    or part.values.device != device
                    or part.values.dtype != dtype
                    or not part.values.is_floating_point()
                    or part.ids.shape != (part.values.shape[0], 4)
                    or part.ids.device != device
                    or (mode != "image_cond" and part.values.shape[0] == 0)
                ):
                    raise ValueError("FLUX3 token widths, devices, dtype and L4 IDs must match the pack")
                count = part.values.shape[0]
                values[mode].append(part.values)
                positions[mode].append(part.ids.to(torch.float32))
                times[mode].append((timesteps[index] if mode == "image" else timesteps.new_zeros(())).expand(count))
                counts[mode] += count
            bounds[mode].append((start, counts[mode]))
    return Flux3Packed(
        {
            mode: torch.cat(values[mode]) if values[mode] else samples[0].image.values.new_zeros(0, widths[mode])
            for mode in MODES
        },
        {
            mode: torch.cat(positions[mode])
            if positions[mode]
            else torch.zeros(0, 4, device=device, dtype=torch.float32)
            for mode in MODES
        },
        {mode: torch.cat(times[mode]) if times[mode] else timesteps.new_zeros(0) for mode in MODES},
        {mode: tuple(bounds[mode]) for mode in MODES},
    )


def validate_packed(packed: Flux3Packed, *, image_channels: int, text_channels: int) -> None:
    """Check complete contiguous bounds and tensor geometry before native projections.

    Host shape checks do not synchronize tensor values. Numerical finiteness and
    sigma domains belong to the data/method boundary. No caller tensor is mutated.
    """
    if any(set(mapping) != set(MODES) for mapping in (packed.data, packed.ids, packed.timesteps, packed.bounds)):
        raise ValueError("FLUX3 packs require exactly image/image_cond/text modes")
    batch = packed.num_samples
    if batch < 1:
        raise ValueError("FLUX3 packs require at least one sample")
    device, dtype = packed.data["image"].device, packed.data["image"].dtype
    for mode in MODES:
        values, ids, time, bounds = packed.data[mode], packed.ids[mode], packed.timesteps[mode], packed.bounds[mode]
        width = text_channels if mode == "text" else image_channels
        if (
            values.ndim != 2
            or values.shape[1] != width
            or ids.shape != (values.shape[0], 4)
            or time.shape != (values.shape[0],)
            or len(bounds) != batch
            or values.dtype != dtype
            or not values.is_floating_point()
            or not time.is_floating_point()
            or any(value.device != device for value in (values, ids, time))
        ):
            raise ValueError("FLUX3 packed widths, IDs, time rows, devices or dtypes differ")
        previous = 0
        for start, end in bounds:
            if (
                type(start) is not int
                or type(end) is not int
                or start != previous
                or end < start
                or (mode != "image_cond" and end == start)
            ):
                raise ValueError("FLUX3 bounds must be contiguous with nonempty image/text samples")
            previous = end
        if previous != values.shape[0]:
            raise ValueError("FLUX3 sample bounds must cover every mode token exactly once")
