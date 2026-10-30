"""Select native Qwen Image 2.1 attention and gated-MLP adapter projections."""

from torch import nn

from unituner.models.qwen_image_2_1.transformer import QwenImage21Transformer


def adapter_projections(model: QwenImage21Transformer) -> dict[str, nn.Linear]:
    """Borrow all seven active bias-free attention/MLP linears in every block."""
    targets: dict[str, nn.Linear] = {}
    for index in range(model.config.num_layers):
        for name in (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "img_mlp.proj",
            "img_mlp.gate_layer",
            "img_mlp.out",
        ):
            path = f"transformer_blocks.{index}.{name}"
            projection = model.get_submodule(path)
            if not isinstance(projection, nn.Linear):
                raise TypeError(f"Qwen Image 2.1 adapter projection must be Linear: {path}")
            targets[path] = projection
    return targets
