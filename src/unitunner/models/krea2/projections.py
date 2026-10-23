"""Select Krea2 native all-linear adapter targets without mutable trainer state."""

from torch import nn

from unitunner.models.krea2.transformer import Krea2Transformer


def adapter_projections(model: Krea2Transformer) -> dict[str, nn.Linear]:
    """Borrow every native Linear, including text fusion, time and final projections.

    Krea2's native selection includes all linear layers. Raw modulation offsets
    and normalization parameters remain excluded. No parameter is replaced.
    """
    return {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
