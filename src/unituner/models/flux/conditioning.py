"""Pack native FLUX latent grids and assemble explicit variant conditions."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class FluxTextFeatures:
    """Carry detached CPU BLD T5 final states and BV CLIP pooled features.

    T5 padding tokens remain active conditioning tokens; no attention mask is
    exported or applied. Both tensors retain their encoder's floating dtype.
    """

    captions: Tensor
    pooled: Tensor


def pack_latents(latents: Tensor) -> Tensor:
    """Return Bx(H/2*W/2)x(C*4) in channel-row-column order.

    Input is floating BCHW with positive even spatial dimensions; gradients pass
    through the reshape/permutation. No VAE normalization is performed.
    """
    if (
        latents.ndim != 4
        or not latents.is_floating_point()
        or min(latents.shape) < 1
        or any(size % 2 for size in latents.shape[-2:])
    ):
        raise ValueError("FLUX packing requires floating BCHW with positive even H/W")
    batch, channels, height, width = latents.shape
    return (
        latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, height * width // 4, channels * 4)
    )


def unpack_latents(tokens: Tensor, height: int, width: int) -> Tensor:
    """Invert packing into BCHW using explicit positive even latent H/W."""
    if (
        tokens.ndim != 3
        or height < 2
        or width < 2
        or height % 2
        or width % 2
        or tokens.shape[1] != height * width // 4
        or tokens.shape[2] % 4
    ):
        raise ValueError("FLUX unpacking requires matching BLC and even latent H/W")
    batch, _, channels = tokens.shape
    return (
        tokens.reshape(batch, height // 2, width // 2, channels // 4, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels // 4, height, width)
    )


def image_ids(batch_size: int, height: int, width: int, *, device: torch.device, reference: bool = False) -> Tensor:
    """Build FP32 Bx(H*W)x3 packed-grid IDs; reference time axis is one.

    Height/width count packed tokens, not VAE pixels or unpacked latent cells.
    """
    if min(batch_size, height, width) < 1:
        raise ValueError("FLUX ID batch and packed grid dimensions must be positive")
    ids = torch.zeros(height, width, 3, device=device)
    ids[..., 0] = float(reference)
    ids[..., 1] = torch.arange(height, device=device)[:, None]
    ids[..., 2] = torch.arange(width, device=device)[None]
    return ids.reshape(1, height * width, 3).expand(batch_size, -1, -1)


def pack_fill_mask(mask: Tensor) -> Tensor:
    """Unshuffle B1HW repaint pixels into Bx64x(H/8)x(W/8).

    H/W are positive multiples of sixteen. One means repaint. Native 8x8
    pixel-channel order is preserved before the subsequent 2x2 latent packing;
    no interpolation, thresholding or dtype conversion occurs.
    """
    if (
        mask.ndim != 4
        or mask.shape[1] != 1
        or not mask.is_floating_point()
        or min(mask.shape[-2:]) < 16
        or any(size % 16 for size in mask.shape[-2:])
    ):
        raise ValueError("FLUX Fill masks require floating B1HW with H/W divisible by sixteen")
    if not torch.all(torch.isfinite(mask) & (mask >= 0) & (mask <= 1)):
        raise ValueError("FLUX Fill mask values must be finite and in [0,1]")
    batch, _, height, width = mask.shape
    return (
        mask.reshape(batch, 1, height // 8, 8, width // 8, 8)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, 64, height // 8, width // 8)
    )


@dataclass(frozen=True, slots=True)
class FluxCondition:
    """Carry packed reference/masked-image tokens, IDs and optional Fill mask.

    Kontext lengths count valid reference prefixes per row, excluding targets.
    Fill mask features are already unshuffled and packed BxLx256 values.
    """

    tokens: Tensor
    ids: Tensor | None = None
    lengths: tuple[int, ...] | None = None
    fill_mask: Tensor | None = None


@dataclass(frozen=True, slots=True)
class FluxInputs:
    """Carry native BLC tokens, BL3 IDs and optional Kontext reference lengths."""

    tokens: Tensor
    ids: Tensor
    control_lengths: tuple[int, ...] | None = None


def build_inputs(
    variant: Literal["dev", "kontext", "fill"], noisy: Tensor, ids: Tensor, condition: FluxCondition | None = None
) -> FluxInputs:
    """Assemble variant channels/tokens without detaching or mutating inputs.

    Reject orphaned or variant-incompatible conditions. Fill mask feature width
    is four times packed target width (256 for native 64-channel tokens).
    """
    if noisy.ndim != 3 or ids.shape != (*noisy.shape[:2], 3):
        raise ValueError("FLUX requires BLC targets with matching BL3 IDs")
    if variant not in ("dev", "kontext", "fill"):
        raise ValueError("FLUX variant must be dev, kontext or fill")
    if condition is None:
        if variant == "fill":
            raise ValueError("FLUX Fill requires masked image and mask features")
        return FluxInputs(noisy, ids)
    if variant == "dev":
        raise ValueError("FLUX Dev does not accept image conditions")
    control = condition.tokens
    if (
        control.ndim != 3
        or control.shape[0] != noisy.shape[0]
        or control.shape[2] != noisy.shape[2]
        or control.device != noisy.device
        or control.dtype != noisy.dtype
    ):
        raise ValueError("FLUX condition batch/width/dtype/device must match target tokens")
    if variant == "fill":
        mask = condition.fill_mask
        if condition.ids is not None or condition.lengths is not None:
            raise ValueError("FLUX Fill cannot consume reference IDs or lengths")
        if (
            control.shape != noisy.shape
            or mask is None
            or mask.shape != (*noisy.shape[:2], 4 * noisy.shape[2])
            or mask.dtype != noisy.dtype
            or mask.device != noisy.device
        ):
            raise ValueError("FLUX Fill requires aligned masked-image tokens and mask channels")
        return FluxInputs(torch.cat((noisy, control, mask), dim=-1), ids)
    if condition.fill_mask is not None:
        raise ValueError("FLUX Kontext cannot consume Fill masks")
    if (
        condition.ids is None
        or condition.ids.shape != (*control.shape[:2], 3)
        or condition.ids.device != ids.device
        or condition.ids.dtype != ids.dtype
    ):
        raise ValueError("FLUX Kontext requires matching reference BL3 IDs")
    lengths = condition.lengths if condition.lengths is not None else (control.shape[1],) * noisy.shape[0]
    if (
        len(lengths) != noisy.shape[0]
        or any(type(length) is not int or not 0 <= length <= control.shape[1] for length in lengths)
        or max(lengths) != control.shape[1]
    ):
        raise ValueError("FLUX Kontext reference lengths must match the padded reference width")
    return FluxInputs(torch.cat((noisy, control), dim=1), torch.cat((ids, condition.ids), dim=1), lengths)
