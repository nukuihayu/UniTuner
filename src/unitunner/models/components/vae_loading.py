"""Read released VAE checkpoints and normalize documented component prefixes."""

from collections.abc import Iterator
from pathlib import Path

from torch import Tensor

from unitunner.artifacts.checkpoint_reading import checkpoint_tensors


def vae_tensors(path: Path, *, prefixes: tuple[str, ...] = ()) -> Iterator[tuple[str, Tensor]]:
    """Read safetensors or Wan PyTorch weights, rejecting prefix collisions."""
    seen: set[str] = set()
    for name, tensor in checkpoint_tensors(path, label="VAE"):
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise ValueError("VAE checkpoint must contain named tensors")
        key = name
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key in seen:
            raise ValueError(f"VAE keys collide after prefix removal: {key}")
        seen.add(key)
        yield key, tensor
