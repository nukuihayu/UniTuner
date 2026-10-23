"""Encode tensor-bearing state as typed JSON nodes and safetensors, never pickle."""

import math
from collections.abc import Mapping

from pydantic import JsonValue
from torch import Tensor

type StateValue = (
    Tensor | str | int | float | bool | list[StateValue] | tuple[StateValue, ...] | dict[str | int, StateValue] | None
)


def encode_state(value: StateValue, tensors: dict[str, Tensor]) -> JsonValue:
    """Validate a third-party state tree and separate each tensor from JSON.

    Preserve tuple/list and integer/string dictionary key distinctions. Reject
    arbitrary objects and nonfinite scalar values; every node has a type tag,
    so user dictionary keys cannot collide with tensor-reference sentinels.
    Caller state is not mutated. Tensor contents are serialized by the caller.
    """
    if isinstance(value, Tensor):
        key = f"state.{len(tensors):08d}"
        tensors[key] = value
        return {"kind": "tensor", "key": key}
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Checkpoint scalar state must be finite")
        return {"kind": "scalar", "value": value}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "values": [encode_state(item, tensors) for item in value],
        }
    if isinstance(value, dict):
        entries: list[JsonValue] = []
        for key, item in value.items():
            if type(key) not in {str, int}:
                raise ValueError("Checkpoint mapping keys must be exact strings or integers")
            entries.append({"key": key, "value": encode_state(item, tensors)})
        return {"kind": "mapping", "entries": entries}
    raise ValueError(f"Unsupported checkpoint state value: {type(value).__name__}")


def decode_state(node: JsonValue, tensors: Mapping[str, Tensor], used: set[str]) -> StateValue:
    """Decode validated tagged JSON, rejecting missing, reused and unexpected tensor refs."""
    if not isinstance(node, dict):
        raise ValueError("Checkpoint node must be a tagged object")
    kind = node.get("kind")
    if not isinstance(kind, str):
        raise ValueError("Checkpoint node kind must be a string")
    if kind == "scalar" and set(node) == {"kind", "value"}:
        value = node["value"]
        if value is None or isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Checkpoint scalar state must be finite")
            return value
    if kind == "tensor" and set(node) == {"kind", "key"}:
        key = node["key"]
        if not isinstance(key, str) or key not in tensors or key in used:
            raise ValueError("Checkpoint tensor references must exist and be unique")
        used.add(key)
        return tensors[key]
    if kind in {"tuple", "list"} and set(node) == {"kind", "values"}:
        values = node["values"]
        if isinstance(values, list):
            decoded = [decode_state(item, tensors, used) for item in values]
            return tuple(decoded) if kind == "tuple" else decoded
    if kind == "mapping" and set(node) == {"kind", "entries"}:
        entries = node["entries"]
        if isinstance(entries, list):
            result: dict[str | int, StateValue] = {}
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {"key", "value"}:
                    raise ValueError("Invalid checkpoint mapping entry")
                key = entry["key"]
                if not isinstance(key, (str, int)) or isinstance(key, bool) or key in result:
                    raise ValueError("Checkpoint mapping keys must be unique strings or integers")
                result[key] = decode_state(entry["value"], tensors, used)
            return result
    raise ValueError("Invalid checkpoint state node")


def string_mapping(value: StateValue) -> dict[str, StateValue]:
    """Narrow one decoded state mapping at the checkpoint boundary."""
    if not isinstance(value, dict):
        raise ValueError("Expected checkpoint mapping")
    result: dict[str, StateValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("Expected checkpoint string keys")
        result[key] = item
    return result
