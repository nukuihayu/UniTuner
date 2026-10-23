"""Publish complete local optimizer phases without pickling Python objects."""

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from unitunner.artifacts.tensors import save_tensors
from unitunner.checkpoints.frozen_state import frozen_tensor_sha256
from unitunner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unitunner.checkpoints.state_tree import StateValue, encode_state

type OptimizerSet = Optimizer | Mapping[str, Optimizer]


def optimizer_roles(optimizer: OptimizerSet) -> dict[str, Optimizer]:
    """Resolve named disjoint optimizer ownership; one optimizer uses role main."""
    roles = {"main": optimizer} if isinstance(optimizer, Optimizer) else dict(optimizer)
    if not roles or any(not name for name in roles):
        raise ValueError("Checkpoint optimizer roles must be nonempty and named")
    parameters = [
        id(parameter) for item in roles.values() for group in item.param_groups for parameter in group["params"]
    ]
    if len(parameters) != len(set(parameters)):
        raise ValueError("Checkpoint optimizers must own disjoint parameters")
    return roles


def optimizer_signature(modules: Mapping[str, nn.Module], optimizer: Optimizer) -> list[list[str]]:
    """Identify ordered optimizer groups by unique qualified parameter roles."""
    names: dict[int, str] = {}
    for module_name, module in modules.items():
        for name, parameter in module.named_parameters():
            if id(parameter) in names:
                raise ValueError("Checkpoint modules must not alias parameter ownership")
            names[id(parameter)] = f"{module_name}.{name}"
    signature: list[list[str]] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        members: list[str] = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in names or identity in seen:
                raise ValueError("Optimizer parameters require unique registered module roles")
            if parameter.grad is not None:
                raise ValueError("Checkpoint only at completed update boundaries after clearing gradients")
            seen.add(identity)
            members.append(names[identity])
        signature.append(members)
    return signature


def save_checkpoint(
    path: Path,
    modules: Mapping[str, nn.Module],
    optimizer: OptimizerSet,
    state: CheckpointCursor,
    identity: ResumeIdentity,
    *,
    cuda_devices: Sequence[int] = (),
    reference_frozen_parameters: bool = False,
) -> None:
    """Atomically save module/optimizer state, CPU/Python/NumPy and explicit CUDA RNG.

    With reference_frozen_parameters, schema3 stores exact content digests for
    frozen parameters in the explicit model role. Restore requires the identical
    already loaded base; all other method modules, including frozen teachers, and
    buffers and trainable/method/optimizer/RNG state remain embedded. Hashing
    trades IO volume for a full base validation scan, without a speed claim.
    Call at a completed update boundary with all model transfers quiescent.
    Modules include adapter/teacher/EMA modules when supplied; their full method
    state contracts remain the caller's responsibility. No scheduler, scaler or
    distributed-rank state is implicitly discovered. Files use safe tensor
    payloads plus a strict tagged JSON tree, never pickle/importable class names.
    """
    roles = optimizer_roles(optimizer)
    signatures = {name: optimizer_signature(modules, item) for name, item in roles.items()}
    numpy_state = np.random.get_state()
    module_states: dict[str | int, StateValue] = {}
    frozen: dict[str, dict[str, str]] = {}
    for name, module in modules.items():
        tensor_state: dict[str | int, StateValue] = dict(module.state_dict().items())
        if reference_frozen_parameters:
            immutable = {
                key: frozen_tensor_sha256(parameter)
                for key, parameter in module.named_parameters()
                if name == "model" and not parameter.requires_grad
            }
            for key in immutable:
                del tensor_state[key]
            frozen[name] = immutable
        module_states[name] = tensor_state
    optimizer_states: dict[str | int, StateValue] = {
        name: dict(item.state_dict().items()) for name, item in roles.items()
    }
    values: dict[str | int, StateValue] = {
        "modules": module_states,
        "optimizer": optimizer_states["main"] if isinstance(optimizer, Optimizer) else optimizer_states,
        "run": {"updates": state.updates, "microbatches": state.microbatches, "samples": state.samples},
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)), *numpy_state[2:]),
        "cuda_rng": {str(device): torch.cuda.get_rng_state(device) for device in cuda_devices},
    }
    tensors: dict[str, torch.Tensor] = {}
    manifest = {
        "schema_version": 1 if isinstance(optimizer, Optimizer) else 2,
        "identity": identity.model_dump(mode="json"),
        "state": encode_state(values, tensors),
    }
    if isinstance(optimizer, Optimizer) and not reference_frozen_parameters:
        manifest["optimizer_type"] = type(optimizer).__module__ + "." + type(optimizer).__qualname__
        manifest["optimizer_signature"] = signatures["main"]
    else:
        manifest["optimizer_types"] = {
            name: type(item).__module__ + "." + type(item).__qualname__ for name, item in roles.items()
        }
        manifest["optimizer_signatures"] = signatures
    if reference_frozen_parameters:
        manifest["schema_version"] = 3
        manifest["frozen_parameters"] = frozen
        manifest["single_optimizer"] = isinstance(optimizer, Optimizer)
    save_tensors(tensors, path, metadata={"unitunner.checkpoint": json.dumps(manifest, allow_nan=False)})
