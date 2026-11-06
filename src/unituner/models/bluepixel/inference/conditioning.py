"""Prepare one RGB BluePixel reference at the chosen output geometry."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def read_reference(path: Path, *, width: int, height: int) -> Tensor:
    """Resize with Lanczos and return CPU BCHW pixels in [-1,1]."""
    with Image.open(path) as image:
        with image.convert("RGB") as rgb, rgb.resize((width, height), Image.Resampling.LANCZOS) as resized:
            pixels = torch.from_numpy(np.array(resized, dtype=np.uint8, copy=True)).permute(2, 0, 1)
    return pixels.unsqueeze(0).float().div_(127.5).sub_(1)
