"""Select explicit optimizer parameters and own reversible trainability/adapter changes."""

import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Literal

import torch
from torch import nn

from unituner.adapters.attachment import AdapterAttachment
from unituner.adapters.config import AdapterConfig
from unituner.adapters.construction import build_branch
from unituner.models.components.quantized import QuantizedLinear


@dataclass(frozen=True, slots=True)
class ParameterGroup:
    """Select named live parameters with an explicit learning-rate multiplier."""

    name: str
    parameters: tuple[nn.Parameter, ...]
    learning_rate_scale: float = 1.0

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.parameters
            or not math.isfinite(self.learning_rate_scale)
            or self.learning_rate_scale <= 0
        ):
            raise ValueError("Parameter group requires a name, parameters and finite positive LR scale")


def validate_parameter_groups(groups: Sequence[ParameterGroup]) -> None:
    """Require named, disjoint, materialized floating trainable parameters."""
    identities = [id(parameter) for group in groups for parameter in group.parameters]
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("Parameter plan must contain nonempty, disjoint groups")
    if len({group.name for group in groups}) != len(groups):
        raise ValueError("Parameter group names must be unique")
    if any(
        not parameter.requires_grad or parameter.is_meta or not parameter.is_floating_point()
        for group in groups
        for parameter in group.parameters
    ):
        raise ValueError("All planned parameters must be materialized floating trainable parameters")


@dataclass(frozen=True, slots=True)
class ParameterPlan:
    """Describe unique live trainable groups without constructing an optimizer."""

    mode: Literal["full", "adapter"]
    groups: tuple[ParameterGroup, ...]
    target_names: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_parameter_groups(self.groups)

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the deterministic optimizer parameter order without copying storage."""
        return tuple(parameter for group in self.groups for parameter in group.parameters)


def select_layer_names(
    names: Iterable[str], *, include_layers: Sequence[str] = (), exclude_layers: Sequence[str] = ()
) -> tuple[str, ...]:
    """Select native paths in inventory order using case-sensitive glob patterns.

    A pattern matches a full path or its module subtree. Empty includes select
    all eligible paths; exclusions win. Validate each pattern against the eligible
    inventory and reject an empty result before changing trainability or RNG.
    LoRA inventories contain projection names; full mode uses parameter names.
    This selects whole native projections, never rows inside fused QKV matrices.
    """
    inventory = tuple(names)

    def matches(name: str, pattern: str) -> bool:
        return fnmatchcase(name, pattern) or fnmatchcase(name, pattern + ".*")

    for pattern in (*include_layers, *exclude_layers):
        if not pattern or pattern != pattern.strip() or not any(matches(name, pattern) for name in inventory):
            raise ValueError(f"Layer pattern {pattern!r} matches no eligible native target")
    selected = tuple(
        name
        for name in inventory
        if (not include_layers or any(matches(name, pattern) for pattern in include_layers))
        and not any(matches(name, pattern) for pattern in exclude_layers)
    )
    if not selected:
        raise ValueError("Layer selection leaves no trainable parameters or adapter targets")
    return selected


class ParameterInstallation:
    """Own trainability and adapter attachment, restoring both on failure or close.

    Native model and Parameter identities stay caller-owned. Adapter state lives
    in a separate ModuleList whose index maps explicitly to plan.target_names.
    Install before the optimizer/recompute/compiler and close in reverse order,
    after all backward calls. Selection uses native names before branch construction;
    full mode selects eligible native parameters and adapter mode selects projections.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        adapter: AdapterConfig | None = None,
        targets: Mapping[str, nn.Linear] | None = None,
        adapter_device: torch.device | None = None,
        adapter_dtype: torch.dtype | None = None,
        include_layers: Sequence[str] = (),
        exclude_layers: Sequence[str] = (),
    ) -> None:
        self.branches = nn.ModuleList()
        self._owners = ExitStack()
        self._closed = False
        native_parameters = tuple(model.parameters())
        quantized = tuple(module for module in model.modules() if isinstance(module, QuantizedLinear))
        if quantized and adapter is None:
            raise ValueError("Quantized bases require adapter training; full parameter training needs floating weights")
        frozen_codes = {id(parameter) for module in quantized for parameter in module.parameters()}
        flags = tuple(parameter.requires_grad for parameter in native_parameters)
        if not native_parameters or any(
            (not parameter.is_floating_point() and id(parameter) not in frozen_codes) or parameter.is_meta
            for parameter in native_parameters
        ):
            raise ValueError("Trainable model parameters must have materialized floating storage")
        if adapter is None and targets:
            raise ValueError("Adapter target projections require adapter configuration")
        if adapter is not None:
            if not targets or len({id(module) for module in targets.values()}) != len(targets):
                raise ValueError("Adapter targets must be nonempty with unique projections")
            native_ids = {id(parameter) for parameter in native_parameters}
            if any(id(module.weight) not in native_ids for module in targets.values()):
                raise ValueError("Adapter targets must belong to the selected model")
        selected_parameters: tuple[nn.Parameter, ...] = ()
        if adapter is None:
            named_parameters = dict(model.named_parameters())
            selected_names = select_layer_names(
                named_parameters, include_layers=include_layers, exclude_layers=exclude_layers
            )
            selected_parameters = tuple(named_parameters[name] for name in selected_names)
        else:
            if targets is None:
                raise ValueError("Adapter targets are required")
            selected_names = select_layer_names(targets, include_layers=include_layers, exclude_layers=exclude_layers)
            targets = {name: targets[name] for name in selected_names}
        for parameter, enabled in zip(native_parameters, flags, strict=True):
            self._owners.callback(parameter.requires_grad_, enabled)
        try:
            model.requires_grad_(False)
            if adapter is None:
                for parameter in selected_parameters:
                    parameter.requires_grad_(True)
                self.plan = ParameterPlan("full", (ParameterGroup("model", selected_parameters),), selected_names)
            else:
                if targets is None:
                    raise ValueError("Adapter targets are required")
                for projection in targets.values():
                    branch = build_branch(projection, adapter, device=adapter_device, dtype=adapter_dtype)
                    attachment = AdapterAttachment(
                        projection, branch, allow_cpu_frozen=adapter_device is not None, fused=adapter.fused
                    )
                    self._owners.callback(attachment.close)
                    self.branches.append(branch)
                self.plan = ParameterPlan(
                    "adapter", (ParameterGroup("adapter", tuple(self.branches.parameters())),), tuple(targets)
                )
        except BaseException:
            self._owners.close()
            raise

    def close(self) -> None:
        """Detach owned branches and restore original native requires_grad flags once."""
        if not self._closed:
            self._closed = True
            self._owners.close()
