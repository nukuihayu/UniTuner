"""Select native Wan block attention and FFN adapter projections."""

from torch import nn

from unitunner.models.wan.config.model import WanConfig
from unitunner.models.wan.transformer import WanTransformer


def adapter_projection_names(config: WanConfig) -> tuple[str, ...]:
    """List the exact block projection inventory without allocating model weights."""
    names = [f"{branch}.{projection}" for branch in ("self_attn", "cross_attn") for projection in ("q", "k", "v", "o")]
    if config.image_context:
        names += ["cross_attn.k_img", "cross_attn.v_img"]
    names += ["ffn.0", "ffn.2"]
    return tuple(f"blocks.{index}.{name}" for index in range(config.num_layers) for name in names)


def adapter_projections(model: WanTransformer) -> dict[str, nn.Linear]:
    """Borrow all block Linear targets used by the native Wan adapter scope.

    Exclude patch/text/time/image embeddings, output head, modulation and norms.
    Include image K/V projections only when the actual block owns that branch.
    Selection changes neither parameter identities nor trainability.
    """
    targets: dict[str, nn.Linear] = {}
    for name in adapter_projection_names(model.config):
        projection = model.get_submodule(name)
        if not isinstance(projection, nn.Linear):
            raise TypeError(f"Wan adapter target must be Linear: {name}")
        targets[name] = projection
    return targets
