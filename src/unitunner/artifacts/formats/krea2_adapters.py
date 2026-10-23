"""Translate native Krea2 projection paths to published Diffusers adapter paths."""

from collections.abc import Iterable
from typing import Literal

from unitunner.artifacts.formats.adapters import adapter_prefixes

_BASIC = {
    "first": "img_in",
    "tmlp.0": "time_embed.linear_1",
    "tmlp.2": "time_embed.linear_2",
    "tproj.1": "time_mod_proj",
    "txtmlp.1": "txt_in.linear_1",
    "txtmlp.3": "txt_in.linear_2",
    "txtfusion.projector": "text_fusion.projector",
    "last.linear": "final_layer.linear",
}
_PROJECTIONS = {
    "attn.wq": "attn.to_q",
    "attn.wk": "attn.to_k",
    "attn.wv": "attn.to_v",
    "attn.gate": "attn.to_gate",
    "attn.wo": "attn.to_out.0",
    "mlp.gate": "ff.gate",
    "mlp.up": "ff.up",
    "mlp.down": "ff.down",
}


def krea2_adapter_prefixes(names: Iterable[str], *, algorithm: Literal["lora", "loha", "lokr"]) -> dict[str, str]:
    """Map all native linears bijectively, rejecting unknown branches/projections.

    Select the canonical to_out.0 spelling. Native parameter paths stay unchanged;
    only published adapter paths use the Diffusers consumer vocabulary. Alpha is
    serialized separately by the shared adapter format boundary.
    """
    paths: dict[str, str] = {}
    for name in names:
        if name in _BASIC:
            paths[name] = _BASIC[name]
            continue
        parts = name.split(".")
        if len(parts) == 4 and parts[0] == "blocks":
            branch, index, projection = "transformer_blocks", parts[1], ".".join(parts[2:])
        elif len(parts) == 5 and parts[0] == "txtfusion" and parts[1] in ("layerwise_blocks", "refiner_blocks"):
            branch, index, projection = f"text_fusion.{parts[1]}", parts[2], ".".join(parts[3:])
        else:
            raise ValueError(f"Unknown Krea2 adapter projection: {name}")
        if not index.isdecimal() or projection not in _PROJECTIONS:
            raise ValueError(f"Unknown Krea2 adapter projection: {name}")
        paths[name] = f"{branch}.{index}.{_PROJECTIONS[projection]}"
    prefixes = adapter_prefixes(paths.values(), algorithm=algorithm)
    return {name: prefixes[path] for name, path in paths.items()}
