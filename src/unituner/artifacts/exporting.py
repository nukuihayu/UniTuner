"""Convert explicit model-native adapter exports for ComfyUI without merging weights."""

from pathlib import Path
from typing import Literal

from unituner.artifacts.formats.adapters import (
    AdapterTargetKeys,
    adapter_key_map,
    adapter_prefixes,
    convert_adapter_keys,
)
from unituner.artifacts.tensors import TensorReader, save_tensors
from unituner.models.minimax_h3.config.model import MiniMaxH3Config
from unituner.models.minimax_h3.projections import adapter_projection_names as minimax_h3_projection_names
from unituner.models.wan.config.model import WanConfig
from unituner.models.wan.projections import adapter_projection_names


def export_comfyui(
    source: Path,
    destination: Path,
    *,
    topology: Path,
    algorithm: Literal["lora", "loha", "lokr"],
    family: Literal["wan", "minimax-h3"] = "wan",
) -> int:
    """Atomically translate a complete native adapter's namespace for ComfyUI.

    Require explicit native topology and algorithm. Validate the complete tensor
    key bijection before writing; retain every tensor value/dtype, including
    alpha. Never infer dotted paths from underscores. The separate consumer file
    uses diffusion_model.*; the canonical source keeps transformer.*/lycoris_*.
    Metadata remains descriptive and is never imported as executable code.
    """
    if destination.resolve() in (source.resolve(), topology.resolve()):
        raise ValueError("Adapter conversion destination must differ from source and topology")
    serialized = topology.read_text(encoding="utf-8")
    match family:
        case "wan":
            names = adapter_projection_names(WanConfig.model_validate_json(serialized))
        case "minimax-h3":
            names = minimax_h3_projection_names(MiniMaxH3Config.model_validate_json(serialized))
        case _:
            raise ValueError("Adapter conversion family must be wan or minimax-h3")
    prefixes = adapter_prefixes(names, algorithm=algorithm)
    canonical = adapter_key_map([AdapterTargetKeys(name, prefixes[name]) for name in names], algorithm=algorithm)
    consumer = adapter_key_map(
        [AdapterTargetKeys(name, "diffusion_model." + name) for name in names], algorithm=algorithm
    )
    keys = {canonical[key]: value for key, value in consumer.items()}
    with TensorReader(source) as reader:
        metadata = reader.metadata()
        if any(key.startswith("ss_") for key in metadata):
            raise ValueError("Adapter conversion requires unprefixed metadata")
        native_layout = f"{family}-native"
        if metadata.get("unituner.adapter.layout", native_layout) != native_layout:
            raise ValueError(f"Adapter conversion requires {native_layout} layout")
        converted = convert_adapter_keys({key: reader.read(key) for key in reader.keys()}, keys)
        metadata["unituner.adapter.layout"] = f"{family}-comfyui"
        save_tensors(converted, destination, metadata=metadata)
    return 0
