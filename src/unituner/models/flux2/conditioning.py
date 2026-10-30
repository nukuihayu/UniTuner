"""Pack FLUX.2 VAE latents and fixed-length text with native four-axis positions."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class Flux2ImageInputs:
    """Carry single-sample BLD image/reference tokens and matching BL4 integer IDs.

    Target tokens precede ordered clean references at time coordinates10,20,... .
    Only the first target_count predictions represent the generated image.
    """

    tokens: Tensor
    ids: Tensor
    target_count: int


def pack_image(latents: Tensor, *, time_coordinate: int = 0) -> tuple[Tensor, Tensor]:
    """Flatten CHW/BCHW packed-VAE latents row-major, preserving gradients.

    VAE2x2 packing is already in channels; no additional spatial patching occurs.
    Return LC/BLC values and L4/BL4 int64(time,row,column,0) IDs on the latent
    device. Neither input storage nor process defaults/RNG are changed.
    """
    if latents.ndim not in (3, 4) or min(latents.shape) < 1 or not latents.is_floating_point():
        raise ValueError("FLUX.2 image latents must be nonempty floating CHW or BCHW tensors")
    if type(time_coordinate) is not int or time_coordinate < 0:
        raise ValueError("FLUX.2 image time_coordinate must be a nonnegative integer")
    height, width = latents.shape[-2:]
    rows, columns = torch.meshgrid(
        torch.arange(height, device=latents.device), torch.arange(width, device=latents.device), indexing="ij"
    )
    ids = torch.stack((torch.full_like(rows, time_coordinate), rows, columns, torch.zeros_like(rows)), dim=-1).reshape(
        -1, 4
    )
    tokens = latents.flatten(-2).transpose(-1, -2)
    return (tokens, ids) if latents.ndim == 3 else (tokens, ids.unsqueeze(0).expand(latents.shape[0], -1, -1))


def pack_text(captions: Tensor) -> tuple[Tensor, Tensor]:
    """Attach native(0,0,0,token) int64 IDs to LD/BLD captions, retaining all padding."""
    if captions.ndim not in (2, 3) or min(captions.shape) < 1 or not captions.is_floating_point():
        raise ValueError("FLUX.2 captions must be nonempty floating LD or BLD tensors")
    ids = torch.zeros((captions.shape[-2], 4), dtype=torch.int64, device=captions.device)
    ids[:, 3] = torch.arange(captions.shape[-2], device=captions.device)
    return (captions, ids) if captions.ndim == 2 else (captions, ids.unsqueeze(0).expand(captions.shape[0], -1, -1))


def unpack_image(tokens: Tensor, *, height: int, width: int) -> Tensor:
    """Restore LC/BLC target tokens to CHW/BCHW using explicit dense geometry.

    Do not infer dimensions from arbitrary IDs or allocate max-coordinate canvases.
    Reference predictions must already be removed by the caller.
    """
    if (
        type(height) is not int
        or type(width) is not int
        or min(height, width) < 1
        or tokens.ndim not in (2, 3)
        or tokens.shape[-2] != height * width
    ):
        raise ValueError("FLUX.2 output tokens must match explicit positive height*width geometry")
    return tokens.transpose(-1, -2).unflatten(-1, (height, width))


def build_image_inputs(target: Tensor, references: tuple[Tensor, ...] = ()) -> Flux2ImageInputs:
    """Append single-sample CHW references without spatial tiling or attention padding.

    Each reference may have independent height/width, sharing target channels,
    device and dtype. Offsets10,20,... preserve native reference ordering. Return
    tensors with singleton batch dimensions; all value gradients remain live.
    """
    if target.ndim != 3:
        raise ValueError("FLUX.2 target must be a single CHW latent")
    target_tokens, target_ids = pack_image(target)
    tokens, ids = [target_tokens], [target_ids]
    for index, reference in enumerate(references):
        if (
            reference.ndim != 3
            or reference.shape[0] != target.shape[0]
            or reference.device != target.device
            or reference.dtype != target.dtype
        ):
            raise ValueError("FLUX.2 references must be CHW with target channels, dtype and device")
        values, positions = pack_image(reference, time_coordinate=10 * (index + 1))
        tokens.append(values)
        ids.append(positions)
    return Flux2ImageInputs(torch.cat(tokens).unsqueeze(0), torch.cat(ids).unsqueeze(0), target_tokens.shape[0])
