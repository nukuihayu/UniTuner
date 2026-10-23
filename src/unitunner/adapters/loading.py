"""Load published projection adapters with strict tensors and reversible ownership."""

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path

import torch
from torch import Tensor, nn

from unitunner.adapters.attachment import AdapterAttachment, LinearBranch
from unitunner.adapters.config import AdapterConfig
from unitunner.adapters.construction import build_branch
from unitunner.adapters.lokr import LoKr
from unitunner.artifacts.formats.adapters import AdapterTargetKeys, adapter_key_map, convert_adapter_keys
from unitunner.artifacts.formats.projection_slices import (
    AdapterProjectionSlice,
    join_projection_state,
    publish_sliced_adapter,
)
from unitunner.artifacts.tensors import iter_tensors
from unitunner.models.components.quantized import linear_dtype


def _validate_tensors(actual: Mapping[str, Tensor], expected: Mapping[str, Tensor]) -> None:
    """Reject incomplete, differently shaped, nonfinite or nonpositive-alpha state."""
    if actual.keys() != expected.keys():
        raise ValueError(
            f"Adapter keys disagree: missing={sorted(expected.keys() - actual.keys())}, extra={sorted(actual.keys() - expected.keys())}"
        )
    for name, tensor in actual.items():
        if tensor.shape != expected[name].shape or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"Adapter tensor shape/dtype/value mismatch for {name}")
        if name.endswith("alpha") and tensor.item() <= 0:
            raise ValueError("Adapter alpha must be positive")


def _load_sliced_branches(
    published: Mapping[str, Tensor],
    config: AdapterConfig,
    targets: Mapping[str, nn.Linear],
    layout: Mapping[str, Sequence[AdapterProjectionSlice]],
) -> tuple[LinearBranch, ...]:
    """Validate every consumer piece, then rejoin native effective branches without merging base weights.

    Caller owns the RNG fork. Expanded LoKr uses the validated expanded rank
    and serialized alpha; it never attempts to recover an arbitrary original
    Kronecker factor gauge. All validation precedes model attachment.
    """
    ordered = {name: layout[name] for name in targets}
    branches: list[LinearBranch] = [build_branch(projection, config) for projection in targets.values()]
    # Validate expanded geometry on metadata tensors, without allocating a
    # second potentially large expanded adapter alongside the imported file.
    shapes = {
        name: torch.empty(value.shape, dtype=value.dtype, device="meta")
        for name, value in nn.ModuleList(branches).state_dict().items()
    }
    expected = publish_sliced_adapter(shapes, ordered, algorithm=config.algorithm)
    _validate_tensors(published, expected)
    for index, (projection, parts) in enumerate(zip(targets.values(), ordered.values(), strict=True)):
        states: list[dict[str, Tensor]] = []
        for part in parts:
            keys = adapter_key_map([AdapterTargetKeys("piece", part.external)], algorithm=config.algorithm)
            states.append({key.removeprefix("piece."): published[external] for key, external in keys.items()})
        original = branches[index]
        expanded = False
        if isinstance(original, LoKr) and original.right_up is not None and len(parts) > 1:
            right_rows = original.right_up.shape[0]
            expanded = any(part.start % right_rows or part.stop % right_rows for part in parts)
        state = join_projection_state(states, algorithm=config.algorithm, expanded_lokr=expanded)
        if expanded:
            branches[index] = LoKr(
                (state["left"].shape[1], state["right_down"].shape[1]),
                (state["left"].shape[0], state["right_up"].shape[0]),
                rank=state["right_down"].shape[0],
                alpha=1.0,
                device=projection.weight.device,
                dtype=linear_dtype(projection),
            )
        branches[index].load_state_dict(state, strict=True)
    return tuple(branches)


class LoadedAdapter:
    """Own inference branches loaded under explicit external prefixes and topology.

    Accept complete factor groups for any nonempty subset of the explicit target
    inventory. Validate every tensor key/shape before attaching any branch, and
    reject unknown or incomplete groups. Never change base
    parameter identity, values, gradient flags or artifact keys. Branch parameters
    are frozen. Caller owns projections and closes after inference/backward; later
    compiler/recompute owners must close first. No runtime trainer dependency exists.
    """

    def __init__(
        self,
        path: Path,
        config: AdapterConfig,
        targets: Mapping[str, nn.Linear],
        *,
        external_prefixes: Mapping[str, str] | None = None,
        projection_layout: Mapping[str, Sequence[AdapterProjectionSlice]] | None = None,
    ) -> None:
        if (external_prefixes is None) == (projection_layout is None):
            raise ValueError("Adapter requires exactly one explicit prefix map or projection layout")
        external_names = external_prefixes if external_prefixes is not None else projection_layout
        if not targets or external_names is None or targets.keys() != external_names.keys():
            raise ValueError("Adapter requires complete explicit external projection prefixes")
        if len({id(projection) for projection in targets.values()}) != len(targets):
            raise ValueError("Adapter projections must be unique")
        published = dict(iter_tensors(path))
        # Partial adapters contain complete factors for a subset of known native
        # projections. Exact key/shape validation below still rejects unknown or
        # incomplete factors, including incomplete fused projection slices.
        published_prefixes = {key.removesuffix(".weight").rpartition(".")[0] for key in published}
        if external_prefixes is not None:
            selected = tuple(name for name in targets if external_prefixes[name] in published_prefixes)
            external_prefixes = {name: external_prefixes[name] for name in selected}
        else:
            if projection_layout is None:
                raise ValueError("Adapter projection layout must be supplied")
            selected = tuple(
                name for name in targets if any(part.external in published_prefixes for part in projection_layout[name])
            )
            projection_layout = {name: projection_layout[name] for name in selected}
        if not selected:
            raise ValueError("Adapter contains no recognized native projections")
        targets = {name: targets[name] for name in selected}
        devices = sorted(
            {
                projection.weight.device.index
                if projection.weight.device.index is not None
                else torch.cuda.current_device()
                for projection in targets.values()
                if projection.weight.device.type == "cuda"
            }
        )
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            if projection_layout is not None:
                branches = _load_sliced_branches(published, config, targets, projection_layout)
                self.branches = nn.ModuleList(branches)
            else:
                if external_prefixes is None:
                    raise ValueError("Adapter projection prefixes must be supplied")
                keys = adapter_key_map(
                    [AdapterTargetKeys(str(index), external_prefixes[name]) for index, name in enumerate(targets)],
                    algorithm=config.algorithm,
                )
                tensors = convert_adapter_keys(published, {external: native for native, external in keys.items()})
                branches = tuple(build_branch(projection, config) for projection in targets.values())
                self.branches = nn.ModuleList(branches)
                _validate_tensors(tensors, self.branches.state_dict())
                self.branches.load_state_dict(tensors, strict=True)
        self.branches.eval().requires_grad_(False)
        self._owners = ExitStack()
        try:
            for projection, branch in zip(targets.values(), branches, strict=True):
                attachment = AdapterAttachment(projection, branch)
                self._owners.callback(attachment.close)
        except BaseException:
            self._owners.close()
            raise

    def close(self) -> None:
        """Detach owned forward replacements once; leave caller modules and weights intact."""
        self._owners.close()
