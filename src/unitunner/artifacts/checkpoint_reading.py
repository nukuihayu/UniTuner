"""Read tensor-only checkpoints without coupling file formats to model topology."""

from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor

from unitunner.artifacts.tensors import iter_tensors


def is_pytorch_checkpoint(path: Path) -> bool:
    """Recognize PyTorch extensions and renamed ZIP/legacy pickle files."""
    if path.is_dir():
        return False
    with path.open("rb") as stream:
        signature = stream.read(9)
    # A safetensors header length can itself start with pickle's 0x80 byte.
    if (
        len(signature) == 9
        and signature[8:9] == b"{"
        and 0 < int.from_bytes(signature[:8], "little") < path.stat().st_size
    ):
        return False
    return path.suffix.lower() in (".pt", ".pth", ".bin") or signature.startswith((b"PK\x03\x04", b"\x80"))


def checkpoint_tensors(path: Path, *, label: str = "Checkpoint") -> Iterator[tuple[str, Tensor]]:
    """Read safe PyTorch state dictionaries or stream safetensors shards on CPU."""
    if is_pytorch_checkpoint(path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or any(
            not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()
        ):
            raise ValueError(f"{label} PyTorch checkpoint must contain a tensor state dict")
        yield from state.items()
    else:
        yield from iter_tensors(path)
