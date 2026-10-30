"""Select native FLUX attention and FFN adapter projections explicitly."""

from torch import nn

from unituner.models.flux.layers import DoubleStreamBlock, SingleStreamBlock
from unituner.models.flux.transformer import FluxTransformer


def adapter_projections(model: FluxTransformer) -> dict[str, nn.Linear]:
    """Borrow ordered double/single targets, excluding modulation and normalization.

    Preserve fused QKV and parallel single-stream projections at native artifact
    keys. No trainability, parameter identity or module lifetime is changed.
    """
    targets: dict[str, nn.Linear] = {}
    for index, block in enumerate(model.double_blocks):
        if not isinstance(block, DoubleStreamBlock):
            raise TypeError("FLUX double-stream topology differs")
        for name in (
            "img_attn.qkv",
            "img_attn.proj",
            "img_mlp.0",
            "img_mlp.2",
            "txt_attn.qkv",
            "txt_attn.proj",
            "txt_mlp.0",
            "txt_mlp.2",
        ):
            projection = block.get_submodule(name)
            if not isinstance(projection, nn.Linear):
                raise TypeError("FLUX adapter projection must be Linear")
            targets[f"double_blocks.{index}.{name}"] = projection
    for index, block in enumerate(model.single_blocks):
        if not isinstance(block, SingleStreamBlock):
            raise TypeError("FLUX single-stream topology differs")
        targets[f"single_blocks.{index}.linear1"] = block.linear1
        targets[f"single_blocks.{index}.linear2"] = block.linear2
    return targets
