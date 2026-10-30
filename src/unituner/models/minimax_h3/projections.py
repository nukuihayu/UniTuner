"""Select native MiniMaxH3 residual-block adapter projections independently of IO."""

from torch import nn

from unituner.models.minimax_h3.config.model import MiniMaxH3Config
from unituner.models.minimax_h3.transformer import MiniMaxH3Transformer


def adapter_projection_names(config: MiniMaxH3Config) -> tuple[str, ...]:
    """List native attention QKV/output and gated-MLP targets for every AV block.

    Retain the original default training inventory. Text refinement, time and
    modality AdaLN projections, patch embeddings and final heads are excluded.
    """
    return tuple(
        f"blocks.{index}.{name}"
        for index in range(config.num_layers)
        for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    )


def adapter_projections(model: MiniMaxH3Transformer) -> dict[str, nn.Linear]:
    """Borrow exact native default targets without changing identities or gradients."""
    targets: dict[str, nn.Linear] = {}
    for name in adapter_projection_names(model.config):
        projection = model.get_submodule(name)
        if not isinstance(projection, nn.Linear):
            raise TypeError(f"MiniMaxH3 adapter target must be Linear: {name}")
        targets[name] = projection
    return targets
