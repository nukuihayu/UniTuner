"""Pack explicit Qwen Image BCLHW latents without ambiguous layer/channel inference."""

import torch
from torch import Tensor


def pack_latents(latents: Tensor) -> Tensor:
    """Pack floating BCLHW into Bx(L*H/2*W/2)x(C*4), channel-row-column patches.

    Layer and channel axes are always explicit, including one-channel layered
    synthetic inputs. No VAE scaling, dtype conversion or mutation occurs;
    reshape/permutation preserve gradients and accept noncontiguous inputs.
    """
    if (
        latents.ndim != 5
        or not latents.is_floating_point()
        or min(latents.shape) < 1
        or any(value % 2 for value in latents.shape[-2:])
    ):
        raise ValueError("Qwen Image packing requires nonempty floating BCLHW with even H/W")
    batch, channels, layers, height, width = latents.shape
    return (
        latents.reshape(batch, channels, layers, height // 2, 2, width // 2, 2)
        .permute(0, 2, 3, 5, 1, 4, 6)
        .reshape(batch, layers * height * width // 4, channels * 4)
    )


def unpack_latents(tokens: Tensor, *, layers: int, height: int, width: int) -> Tensor:
    """Invert native patches to BCLHW using explicit latent geometry, preserving gradients."""
    if (
        tokens.ndim != 3
        or min(tokens.shape) < 1
        or not tokens.is_floating_point()
        or min(layers, height, width) < 1
        or height % 2
        or width % 2
        or tokens.shape[1] != layers * height * width // 4
        or tokens.shape[2] % 4
    ):
        raise ValueError("Qwen Image unpacking requires BNC and exact positive layer/even spatial geometry")
    batch, _, channels = tokens.shape
    return (
        tokens.reshape(batch, layers, height // 2, width // 2, channels // 4, 2, 2)
        .permute(0, 4, 1, 2, 5, 3, 6)
        .reshape(batch, channels // 4, layers, height, width)
    )


def batch_context(captions: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
    """Pad nonempty LD contexts after encoding, returning BLD and boolean prefix mask."""
    if not captions or any(value.ndim != 2 or min(value.shape) < 1 for value in captions):
        raise ValueError("Qwen Image requires one nonempty LD context per sample")
    first = captions[0]
    if not first.is_floating_point() or any(
        value.shape[1] != first.shape[1] or value.dtype != first.dtype or value.device != first.device
        for value in captions
    ):
        raise ValueError("Qwen Image contexts must share floating width/dtype/device")
    longest = max(value.shape[0] for value in captions)
    values = torch.stack([torch.nn.functional.pad(value, (0, 0, 0, longest - value.shape[0])) for value in captions])
    mask = (
        torch.arange(longest, device=first.device)[None]
        < torch.tensor([value.shape[0] for value in captions], device=first.device)[:, None]
    )
    return values, mask
