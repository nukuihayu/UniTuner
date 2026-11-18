"""Select MOSS main and caption-refiner adapter projections explicitly."""

from torch import nn

from unituner.models.moss_image.layers import MossImageBlock, TextBlock
from unituner.models.moss_image.transformer import MossImageTransformer


def adapter_projections(model: MossImageTransformer) -> dict[str, nn.Linear]:
    """Borrow attention/FFN linears in native order, excluding norms and modulation.

    Include caption refiners, as in the original MOSS target policy. Native text
    QKV and main hidden/gate FFN remain fused. No parameter or forward is mutated.
    """
    targets: dict[str, nn.Linear] = {}
    for index, block in enumerate(model.text_embedder.blocks):
        if not isinstance(block, TextBlock):
            raise TypeError("MOSS caption block topology differs")
        prefix = f"text_embedder.blocks.{index}"
        for name in ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down"):
            projection = block.get_submodule(name)
            if not isinstance(projection, nn.Linear):
                raise TypeError("MOSS caption projection must be Linear")
            targets[f"{prefix}.{name}"] = projection
    for index, block in enumerate(model.layers):
        if not isinstance(block, MossImageBlock):
            raise TypeError("MOSS main block topology differs")
        prefix = f"layers.{index}.proj"
        for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
            projection = block.proj.get_submodule(name)
            if not isinstance(projection, nn.Linear):
                raise TypeError("MOSS main projection must be Linear")
            targets[f"{prefix}.{name}"] = projection
    return targets
