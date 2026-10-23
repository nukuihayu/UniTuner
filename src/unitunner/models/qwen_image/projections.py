"""Select Qwen Image native attention and FFN adapter targets."""

from torch import nn

from unitunner.models.qwen_image.transformer import QwenImageTransformer


def adapter_projections(model: QwenImageTransformer) -> dict[str, nn.Linear]:
    """Borrow active stream projections, excluding final unused text output/FFN.

    Retain final text K/V inputs because they affect image attention. Its text
    query affects only discarded text outputs and is also excluded.
    Modulation, time projections and normalization are not adapter targets.
    Published native module paths remain the artifact conversion boundary.
    """
    targets: dict[str, nn.Linear] = {}
    for index in range(model.config.num_layers):
        names = [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_out.0",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
        ]
        if index < model.config.num_layers - 1:
            names += ["attn.add_q_proj", "attn.to_add_out", "txt_mlp.net.0.proj", "txt_mlp.net.2"]
        for name in names:
            path = f"transformer_blocks.{index}.{name}"
            projection = model.get_submodule(path)
            if not isinstance(projection, nn.Linear):
                raise TypeError(f"Qwen Image adapter target must be Linear: {path}")
            targets[path] = projection
    return targets
