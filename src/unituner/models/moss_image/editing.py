"""Moss-Image-Edit reference geometry and isolated attention (ComfyUI PR #17296)."""

import torch
from torch import Tensor
from torch.nn import functional as F

from unituner.execution.attention import scaled_dot_product_attention


def reference_rope(shapes: tuple[tuple[int, int], ...], bands: Tensor) -> Tensor:
    """Place reference i at (i+1, 0, centered row, centered column)."""
    positions = []
    for index, (height, width) in enumerate(shapes):
        rows, columns = torch.meshgrid(
            torch.arange(height, device=bands.device), torch.arange(width, device=bands.device), indexing="ij"
        )
        ids = torch.stack((torch.full_like(rows, index + 1), torch.zeros_like(rows), rows, columns), -1).float()
        ids[..., 2] -= (height - 1) / 2
        ids[..., 3] -= (width - 1) / 2
        positions.append(ids.flatten(0, 1))
    phase = torch.cat(positions).unsqueeze(-1) * bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)[None]


def attend_body(
    query: Tensor, key: Tensor, value: Tensor, lengths: tuple[int, ...], reference_kv: tuple[Tensor, Tensor]
) -> Tensor:
    """Attend valid target/text prefixes plus all references, excluding text padding."""
    rk, rv = reference_kv
    rows = []
    for index, size in enumerate(lengths):
        keys = torch.cat((key[index : index + 1, :size], rk[index : index + 1]), 1)
        values = torch.cat((value[index : index + 1, :size], rv[index : index + 1]), 1)
        out = (
            scaled_dot_product_attention(
                query[index : index + 1, :size].transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
            )
            .transpose(1, 2)
            .flatten(2)
        )
        rows.append(F.pad(out, (0, 0, 0, query.shape[1] - size)))
    return torch.cat(rows)
