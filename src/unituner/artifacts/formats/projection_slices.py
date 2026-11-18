"""Split and join adapter factors at explicit consumer output-row boundaries."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from unituner.artifacts.formats.adapters import AdapterTargetKeys, adapter_key_map, convert_adapter_keys


@dataclass(frozen=True, slots=True)
class AdapterProjectionSlice:
    """Name a consumer projection covering contiguous native output rows [start, stop)."""

    external: str
    start: int
    stop: int


def validate_projection_slices(parts: Sequence[AdapterProjectionSlice], out_features: int) -> None:
    """Require complete ordered row coverage and distinct nonempty external names."""
    position = 0
    names: set[str] = set()
    for part in parts:
        if part.start != position or part.stop <= part.start or not part.external or part.external in names:
            raise ValueError("Adapter projection slices require ordered complete rows and unique nonempty names")
        position = part.stop
        names.add(part.external)
    if position != out_features:
        raise ValueError("Adapter projection slices do not cover the native output width")


def split_projection_state(
    state: Mapping[str, Tensor],
    parts: Sequence[AdapterProjectionSlice],
    *,
    algorithm: Literal["lora", "loha", "lokr"],
) -> tuple[dict[str, Tensor], ...]:
    """Split native factors without changing training parameters or materializing a dense delta.

    LoRA/LoHa slice output factors and repeat input factors and alpha. LoKr
    slices its left factor when every boundary aligns with the right output
    width. Otherwise use kron(L, U) @ kron(I, D), keeping a singleton left
    factor. The expanded rank is in_left*rank, so alpha is multiplied by
    in_left too. Expansion is exact algebra, with ordinary floating GEMM
    rounding; it increases artifact storage only for unaligned partitions.
    Returned views/expanded tensors remain on the caller device and retain
    gradients; callers materialize immutable contiguous tensors when saving.
    """
    if algorithm == "lora":
        rows = ("up",)
        out_features = state["up"].shape[0]
    elif algorithm == "loha":
        rows = ("a1", "a2")
        out_features = state["a1"].shape[0]
    else:
        rows = ("left",)
        out_features = state["left"].shape[0] * state["right_up"].shape[0]
    validate_projection_slices(parts, out_features)
    if len(parts) == 1:
        return (dict(state),)
    if algorithm != "lokr":
        return tuple(
            {name: value[part.start : part.stop] if name in rows else value for name, value in state.items()}
            for part in parts
        )
    left, up, down = state["left"], state["right_up"], state["right_down"]
    right_rows = up.shape[0]
    if all(part.start % right_rows == 0 and part.stop % right_rows == 0 for part in parts):
        return tuple({**state, "left": left[part.start // right_rows : part.stop // right_rows]} for part in parts)
    expanded_up = torch.kron(left.contiguous(), up.contiguous())
    expanded_down = torch.kron(torch.eye(left.shape[1], dtype=down.dtype, device=down.device), down.contiguous())
    return tuple(
        {
            "left": left.new_ones(1, 1),
            "right_up": expanded_up[part.start : part.stop],
            "right_down": expanded_down,
            "alpha": state["alpha"] * left.shape[1],
        }
        for part in parts
    )


def join_projection_state(
    states: Sequence[Mapping[str, Tensor]], *, algorithm: Literal["lora", "loha", "lokr"], expanded_lokr: bool = False
) -> dict[str, Tensor]:
    """Rejoin validated pieces, rejecting inconsistent shared factors and scales.

    Native LoRA/LoHa and aligned LoKr factors are recovered bit for bit. An
    expanded LoKr stays in its exact expanded representation; no SVD, dense
    reconstruction or arbitrary gauge recovery is performed.
    """
    if not states:
        raise ValueError("Adapter projection states must be nonempty")
    first = states[0]
    if len(states) == 1:
        return dict(first)
    rows = ("up",) if algorithm == "lora" else ("a1", "a2") if algorithm == "loha" else ("left",)
    if expanded_lokr:
        if algorithm != "lokr":
            raise ValueError("Only LoKr projection states support Kronecker expansion")
        rows = ("right_up",)
    for state in states:
        if state.keys() != first.keys():
            raise ValueError("Adapter projection factor inventories disagree")
        for name in first.keys() - set(rows):
            if not torch.equal(state[name], first[name]):
                raise ValueError(f"Adapter projection slices disagree on shared factor {name}")
    return {
        name: torch.cat([state[name] for state in states], dim=0) if name in rows else value
        for name, value in first.items()
    }


def publish_sliced_adapter(
    tensors: Mapping[str, Tensor],
    layout: Mapping[str, Sequence[AdapterProjectionSlice]],
    *,
    algorithm: Literal["lora", "loha", "lokr"],
) -> dict[str, Tensor]:
    """Publish complete indexed branch state under explicit consumer row projections."""
    internal = adapter_key_map(
        [AdapterTargetKeys(str(index), str(index)) for index in range(len(layout))], algorithm=algorithm
    )
    # Validate the entire inventory before constructing any partial publication.
    convert_adapter_keys(tensors, {key: key for key in internal})
    published: dict[str, Tensor] = {}
    for index, parts in enumerate(layout.values()):
        state = {key.removeprefix(f"{index}."): value for key, value in tensors.items() if key.startswith(f"{index}.")}
        for part, piece in zip(parts, split_projection_state(state, parts, algorithm=algorithm), strict=True):
            keys = adapter_key_map([AdapterTargetKeys("piece", part.external)], algorithm=algorithm)
            converted = convert_adapter_keys({"piece." + key: value for key, value in piece.items()}, keys)
            if published.keys() & converted.keys():
                raise ValueError("Adapter consumer projections must be globally bijective")
            published.update(converted)
    return published
