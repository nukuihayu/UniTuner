"""Flatten native image-only Qwen2.1 latents without previous-generation patch packing."""

from dataclasses import dataclass

from torch import Tensor


def pack_latents(latents: Tensor) -> Tensor:
    """View floating BC1HW pixels as B(HW)C row-major tokens, retaining gradients."""
    if latents.ndim != 5 or min(latents.shape) < 1 or latents.shape[2] != 1 or not latents.is_floating_point():
        raise ValueError("Qwen Image2.1 packing requires nonempty floating BC1HW")
    return latents.flatten(2).transpose(1, 2)


def unpack_latents(tokens: Tensor, *, height: int, width: int) -> Tensor:
    """Reshape row-major B(HW)C velocity tokens to BC1HW without changing values."""
    if (
        tokens.ndim != 3
        or min(tokens.shape) < 1
        or not tokens.is_floating_point()
        or min(height, width) < 1
        or tokens.shape[1] != height * width
    ):
        raise ValueError("Qwen Image2.1 unpacking requires floating B(HW)C matching positive H/W")
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], 1, height, width)


@dataclass(frozen=True, slots=True)
class QwenImage21TextFeatures:
    """Carry detached unpadded LD text, insertion slots and reference (H,W) grids.

    Slots count retained text tokens before each removed contiguous vision span.
    Shapes use VAE factor16 grids, in the caller's reference order. Metadata
    stays on the host; features retain the encoder's floating dtype/device.
    """

    features: Tensor
    image_slots: tuple[int, ...]
    image_shapes: tuple[tuple[int, int], ...]
