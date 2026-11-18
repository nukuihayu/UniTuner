"""Identify native Z-Image attention/FFN projection geometry explicitly."""

from torch import nn

from unituner.models.z_image.layers import ZImageTransformerBlock
from unituner.models.z_image.transformer import ZImageTransformer


def adapter_projections(model: ZImageTransformer) -> dict[str, nn.Linear]:
    """Return ordered main-block attention/FFN projections under native artifact keys.

    Exclude refiners and modulation to preserve the existing default target policy.
    The returned modules are borrowed; no trainability or forward mutation occurs.
    """
    targets: dict[str, nn.Linear] = {}
    for index, block in enumerate(model.layers):
        if not isinstance(block, ZImageTransformerBlock):
            raise TypeError("Z-Image main block topology differs")
        prefix = f"layers.{index}"
        targets[f"{prefix}.attention.to_q"] = block.attention.to_q
        targets[f"{prefix}.attention.to_k"] = block.attention.to_k
        targets[f"{prefix}.attention.to_v"] = block.attention.to_v
        output = block.attention.to_out[0]
        if not isinstance(output, nn.Linear):
            raise TypeError("Z-Image output projection must be Linear")
        targets[f"{prefix}.attention.to_out.0"] = output
        targets[f"{prefix}.feed_forward.w1"] = block.feed_forward.w1
        targets[f"{prefix}.feed_forward.w2"] = block.feed_forward.w2
        targets[f"{prefix}.feed_forward.w3"] = block.feed_forward.w3
    return targets
