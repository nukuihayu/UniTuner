"""Convert explicit adapter tensor keys and metadata without inferring model topology."""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from torch import Tensor


@dataclass(frozen=True, slots=True)
class AdapterTargetKeys:
    """Map one semantic projection between new internal and published external names."""

    internal: str
    external: str


def adapter_key_map(
    targets: Sequence[AdapterTargetKeys],
    *,
    algorithm: Literal["lora", "loha", "lokr", "lokr-full"],
) -> dict[str, str]:
    """Build a bijection, rejecting lossy underscore/path collisions.

    Model bindings supply exact external prefixes (for example transformer.* or
    lycoris_*). No model is detected from class names or tensor shapes. Alpha is
    always a serialized scale tensor; no Python import path is made executable.
    """
    suffixes = {
        "lora": {"down": "lora_down.weight", "up": "lora_up.weight"},
        "loha": {"a1": "hada_w1_a", "b1": "hada_w1_b", "a2": "hada_w2_a", "b2": "hada_w2_b"},
        "lokr": {"left": "lokr_w1", "right_up": "lokr_w2_a", "right_down": "lokr_w2_b"},
        "lokr-full": {"left": "lokr_w1", "right": "lokr_w2"},
    }[algorithm]
    suffixes = {**suffixes, "alpha": "alpha"}
    mapping: dict[str, str] = {}
    destinations: set[str] = set()
    for target in targets:
        if not target.internal or not target.external:
            raise ValueError("Adapter projection prefixes must be nonempty")
        for internal, external in suffixes.items():
            source, destination = f"{target.internal}.{internal}", f"{target.external}.{external}"
            if source in mapping or destination in destinations:
                raise ValueError("Adapter key map must be bijective; duplicate projection or external prefix")
            mapping[source] = destination
            destinations.add(destination)
    return mapping


def convert_adapter_keys(tensors: Mapping[str, Tensor], keys: Mapping[str, str]) -> dict[str, Tensor]:
    """Convert all and only mapped keys, retaining tensor values, identity and dtype.

    Missing/extra tensors and non-bijective destinations are errors. Returned
    tensors share caller storage; no weights are materialized, merged or cast.
    """
    missing, extra = keys.keys() - tensors.keys(), tensors.keys() - keys.keys()
    if missing or extra:
        raise ValueError(f"Adapter keys disagree: missing={sorted(missing)}, extra={sorted(extra)}")
    if len(set(keys.values())) != len(keys):
        raise ValueError("Adapter key destinations must be unique")
    return {keys[key]: tensor for key, tensor in tensors.items()}


def adapter_metadata(
    *,
    algorithm: Literal["lora", "loha", "lokr"],
    base_model: str,
    rank: int,
    alpha: float,
    lora_module_identifier: str | None = None,
) -> dict[str, str]:
    """Describe the model and adapter with unprefixed string metadata fields.

    The caller supplies the consumer-tested LoRA identifier for its model family.
    LyCORIS types have their own explicit algo declaration. These external labels
    are never dynamically imported. Actual ComfyUI loading remains a separate
    provider acceptance step, not a consequence of this metadata schema.
    """
    import_identifier = lora_module_identifier if algorithm == "lora" else "lycoris.kohya"
    if not import_identifier or not base_model or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Adapter metadata requires base model, positive rank and a format identifier")
    arguments = "{}" if algorithm == "lora" else '{"algo":"' + algorithm + '"}'
    return {
        "base_model_version": base_model,
        "network_module": import_identifier,
        "network_dim": str(rank),
        "network_alpha": str(alpha),
        "network_args": arguments,
    }


def adapter_prefixes(
    names: Iterable[str],
    *,
    algorithm: Literal["lora", "loha", "lokr"],
    namespace: Literal["transformer", "diffusion_model"] = "transformer",
) -> dict[str, str]:
    """Map explicit consumer projection paths to algorithm-specific namespaces.

    The transformer namespace uses transformer.<path> for LoRA and
    lycoris_<underscored-path> for LoHa/LoKr. The diffusion_model namespace
    uses dotted native paths for every algorithm, as required by ComfyUI's
    FLUX3, MOSS, Wan and H3 loaders. Factor suffixes still identify the algorithm.
    Callers must supply consumer paths: this function does not split fused QKV
    or translate native model topology to Diffusers. Prefixes alone do not
    establish ComfyUI support for a model. Reject empty and colliding paths.
    """
    if algorithm not in ("lora", "loha", "lokr"):
        raise ValueError("Adapter format requires lora/loha/lokr")
    if namespace not in ("transformer", "diffusion_model"):
        raise ValueError("Adapter namespace requires transformer/diffusion_model")
    result = {
        name: "diffusion_model." + name
        if namespace == "diffusion_model"
        else "transformer." + name
        if algorithm == "lora"
        else "lycoris_" + name.replace(".", "_")
        for name in names
    }
    if len(set(result.values())) != len(result) or any(not name for name in result):
        raise ValueError("Adapter projection prefixes must be nonempty and bijective")
    return result
