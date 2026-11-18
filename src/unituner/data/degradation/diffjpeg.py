"""CPU float JPEG with the non-differentiable DiffJPEG math used by HYPIR.

Adapted from Michael R. Lomnitz's MIT-licensed DiffJPEG (2021), via BasicSR.
See LICENSE-DiffJPEG.txt. Uses 4:2:0, floating quantization tables, nearest
chroma reconstruction and zero padding to 16, including at image borders.
"""

import math
from functools import lru_cache

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F


@lru_cache(maxsize=1)
def _constants() -> tuple[torch.Tensor, ...]:
    """Build immutable CPU constants with upstream float32 rounding points."""
    rgb = torch.tensor([[0.299, -0.168736, 0.5], [0.587, -0.331264, -0.418688], [0.114, 0.5, -0.081312]])
    inverse = torch.tensor([[1.0, 1.0, 1.0], [0.0, -0.344136, 1.772], [1.402, -0.714136, 0.0]])
    y = torch.tensor(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ],
        dtype=torch.float32,
    ).T
    c = torch.full((8, 8), 99.0)
    c[:4, :4] = torch.tensor([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]]).T
    # NumPy double cosine before casting matches upstream constant construction.
    x, y_pos, u, v = np.indices((8, 8, 8, 8))
    basis = torch.from_numpy(
        (np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y_pos + 1) * v * np.pi / 16)).astype(np.float32)
    )
    alpha = np.outer(np.array([1 / math.sqrt(2)] + [1] * 7), np.array([1 / math.sqrt(2)] + [1] * 7))
    return (
        rgb,
        inverse,
        y,
        c,
        basis,
        torch.from_numpy((alpha * 0.25).astype(np.float32)),
        torch.from_numpy(alpha.astype(np.float32)),
    )


def _component(value: torch.Tensor, table: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Round-trip one plane through 8x8 DCT, quantization and inverse DCT."""
    _, height, width = value.shape
    blocks = value.reshape(1, height // 8, 8, width // 8, 8).permute(0, 1, 3, 2, 4).reshape(1, -1, 8, 8)
    *_, basis, forward_scale, alpha = _constants()
    coefficients = forward_scale * torch.tensordot(blocks - 128, basis, dims=2)
    # Tensor factor follows HYPIR's per-image quality tensor, including float32.
    quantization = table * factor
    coefficients = (coefficients / quantization).round() * quantization
    blocks = 0.25 * torch.tensordot(coefficients * alpha, basis.permute(2, 3, 0, 1).contiguous(), dims=2) + 128
    return blocks.reshape(1, height // 8, width // 8, 8, 8).permute(0, 1, 3, 2, 4).reshape(1, height, width)


@torch.no_grad()
@torch.autocast("cpu", enabled=False)
def diffjpeg(image: NDArray[np.float32], quality: float) -> NDArray[np.float32]:
    """Round-trip HWC float32 RGB through HYPIR-compatible float JPEG."""
    if not 0 < quality < 100:
        raise ValueError("DiffJPEG quality must be strictly between 0 and 100")
    rgb, inverse, y_table, c_table, *_ = _constants()
    value = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
    height, width = value.shape[-2:]
    value = F.pad(value, (0, -width % 16, 0, -height % 16))
    ycc = torch.tensordot((value * 255).permute(0, 2, 3, 1), rgb, dims=1) + torch.tensor([0.0, 128.0, 128.0])
    # Calculate quality with float32 tensor arithmetic, as in official HYPIR.
    quality_tensor = torch.tensor(quality, dtype=torch.float32)
    factor = (5000 / quality_tensor if quality < 50 else 200 - quality_tensor * 2) / 100
    y = _component(ycc[..., 0], y_table, factor)
    chroma = F.avg_pool2d(ycc[..., 1:].permute(0, 3, 1, 2), 2, stride=2)
    cb = _component(chroma[:, 0], c_table, factor).repeat_interleave(2, 1).repeat_interleave(2, 2)
    cr = _component(chroma[:, 1], c_table, factor).repeat_interleave(2, 1).repeat_interleave(2, 2)
    decoded = torch.stack((y, cb, cr), dim=-1) + torch.tensor([0.0, -128.0, -128.0])
    decoded = torch.tensordot(decoded, inverse, dims=1).clamp(0, 255) / 255
    return decoded[0, :height, :width].contiguous().numpy()
