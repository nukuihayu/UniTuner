"""Encode independent Wan target/clean frames with explicit native temporal coordinates."""

from dataclasses import dataclass

import torch
from torch import Tensor

from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.models.wan.conditioning import validate_wan_vae


@dataclass(frozen=True, slots=True)
class WanSingleFrameLayout:
    """Pin sorted native temporal indices, original clean-image order and target slot."""

    frame_indices: tuple[int, ...]
    clean_order: tuple[int, ...]
    target_position: int


@dataclass(frozen=True, slots=True)
class WanSingleFrameConditions:
    """Own B16FHW training latents, B20FHW I2V conditions and host frame coordinates."""

    latents: Tensor
    image_condition: Tensor
    layout: WanSingleFrameLayout


def build_single_frame_layout(clean_indices: tuple[int, ...], target_index: int) -> WanSingleFrameLayout:
    """Bind one/two clean images to unique native coordinates in[0,1024).

    Sorting moves each image together with its coordinate. Reversed input
    order cannot silently assign a clean image to another temporal position.
    """
    indices = (*clean_indices, target_index)
    if not 1 <= len(clean_indices) <= 2 or any(type(index) is not int or not 0 <= index < 1024 for index in indices):
        raise ValueError("Wan single-frame requires one/two clean indices and a target index in[0,1024)")
    if len(set(indices)) != len(indices):
        raise ValueError("Wan single-frame clean/target indices must be distinct")
    ordered = tuple(sorted(indices))
    return WanSingleFrameLayout(
        ordered, tuple(sorted(range(len(clean_indices)), key=clean_indices.__getitem__)), ordered.index(target_index)
    )


def encode_single_frame_conditions(
    vae: QwenImageVAE,
    target_pixels: Tensor,
    clean_pixels: Tensor,
    *,
    clean_indices: tuple[int, ...],
    target_index: int,
) -> WanSingleFrameConditions:
    """Encode each RGB image independently and assemble native temporal slots.

    Inputs are same-device FP32 BC1HW target and BCKHW clean RGB pixels in
    [-1,1], K=1/2. Encode per row/frame with isolated causal histories. Target
    condition uses the encoded normalized-zero image, not pixel-black (-1).
    Clean conditions include four visibility channels of ones and the clean
    mean; target visibility is zero. These are native I2V inputs, never RCM.

    Preserve gradients to target/clean pixels and all active VAE parameters.
    No global or shape-only neutral-image cache can leak across VAE weights,
    device or precision changes. Caller owns no-grad/frozen state and residency.
    """
    layout = build_single_frame_layout(clean_indices, target_index)
    validate_wan_vae(vae)
    if (
        target_pixels.ndim != 5
        or clean_pixels.ndim != 5
        or target_pixels.shape[1:3] != (3, 1)
        or clean_pixels.shape[:2] != target_pixels.shape[:2]
        or clean_pixels.shape[2] != len(clean_indices)
        or clean_pixels.shape[-2:] != target_pixels.shape[-2:]
        or min(target_pixels.shape) < 1
        or target_pixels.dtype != torch.float32
        or clean_pixels.dtype != torch.float32
        or clean_pixels.device != target_pixels.device
        or any(
            not torch.isfinite(value).all() or value.min() < -1 or value.max() > 1
            for value in (target_pixels, clean_pixels)
        )
    ):
        raise ValueError("Wan single-frame requires matching finite FP32 RGB BC1HW/BCKHW pixels in[-1,1]")
    target_latents = torch.cat([vae.encode(row.unsqueeze(0)) for row in target_pixels])
    clean_latents = tuple(
        torch.cat([vae.encode(row[:, index : index + 1].unsqueeze(0)) for row in clean_pixels])
        for index in layout.clean_order
    )
    neutral = vae.encode(torch.zeros_like(target_pixels[:1])).expand(target_pixels.shape[0], -1, -1, -1, -1)
    latents, conditions = [], []
    clean_position = 0
    for position in range(len(layout.frame_indices)):
        if position == layout.target_position:
            latents.append(target_latents)
            conditions.append(torch.cat((torch.zeros_like(neutral[:, :4]), neutral), dim=1))
        else:
            values = clean_latents[clean_position]
            latents.append(values)
            conditions.append(torch.cat((torch.ones_like(values[:, :4]), values), dim=1))
            clean_position += 1
    return WanSingleFrameConditions(torch.cat(latents, dim=2), torch.cat(conditions, dim=2), layout)
