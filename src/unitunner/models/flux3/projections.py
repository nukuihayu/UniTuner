"""Describe native FLUX3 attention/FFN adapter targets without installing resources."""

from torch import nn

from unitunner.models.flux3.conditioning import MODES
from unitunner.models.flux3.transformer import Flux3Transformer


def adapter_projections(model: Flux3Transformer) -> dict[str, nn.Linear]:
    """Borrow all early-mode/shared projections, excluding time/modulation/norm weights."""
    prefixes = {
        "image": "content_mode_blocks.image",
        "image_cond": "content_mode_blocks.image_cond",
        "text": "txt_mode_blocks",
    }
    groups = [(prefixes[mode], model.early_blocks(mode)) for mode in MODES]
    groups.append(("single_blocks", model.single_blocks))
    targets = {}
    for prefix, blocks in groups:
        for index, block in enumerate(blocks):
            for name in ("q_proj", "k_proj", "v_proj", "mlp_in", "attn_out", "mlp_out"):
                projection = block.get_submodule(name)
                if not isinstance(projection, nn.Linear):
                    raise TypeError("FLUX3 adapter projection must be Linear")
                targets[f"{prefix}.{index}.{name}"] = projection
    return targets
