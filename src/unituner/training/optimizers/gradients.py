"""Normalize dense optimizer gradients without one CUDA launch per parameter."""

import torch
from torch import (
    Tensor,
    _foreach_div_,  # pyright: ignore[reportPrivateImportUsage] -- Torch omits this supported operator from __all__.
)


def normalize_gradients(gradients: list[Tensor], samples: int) -> None:
    """Divide gradients in place by actual samples, preserving native division math.

    Torch exposes no public batched in-place scalar division. The private foreach
    operator is isolated here and limited to the declared Torch2.13 dependency.
    Sparse gradients retain the scalar operation. No gradients are concatenated,
    copied to host or detached from optimizer-owned storage. Division by one is
    an exact identity and needs no launch. Caller validates finite gradients.
    """
    if samples <= 0:
        raise ValueError("Gradient normalization requires positive samples")
    if samples == 1:
        return
    if torch.__version__.split("+")[0].split(".")[:2] != ["2", "13"]:
        raise RuntimeError("Batched gradient division requires supported Torch2.13")
    if any(gradient.is_sparse for gradient in gradients):
        for gradient in gradients:
            gradient.div_(samples)
    elif gradients:
        _foreach_div_(gradients, samples)
