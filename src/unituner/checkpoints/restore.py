"""Restore local state into a freshly constructed session without installing execution policies."""

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from torch import Tensor, nn
from torch.optim import Optimizer

from unituner.artifacts.tensors import TensorReader
from unituner.checkpoints.saving import OptimizerSet, optimizer_roles, optimizer_signature
from unituner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unituner.checkpoints.state_tree import StateValue, decode_state, frozen_tensor_sha256, string_mapping


class CheckpointManifest(BaseModel):
    """Validate the state envelope before touching any caller module or RNG."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal[1]
    identity: ResumeIdentity
    optimizer_type: str
    optimizer_signature: list[list[str]]
    state: JsonValue


class MultiOptimizerManifest(BaseModel):
    """Validate named optimizer phases in a native multi-optimizer checkpoint."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal[2]
    identity: ResumeIdentity
    optimizer_types: dict[str, str]
    optimizer_signatures: dict[str, list[list[str]]]
    state: JsonValue


class FrozenReferenceManifest(BaseModel):
    """Require exact preloaded frozen parameter content, retaining all mutable state."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal[3]
    identity: ResumeIdentity
    optimizer_types: dict[str, str]
    optimizer_signatures: dict[str, list[list[str]]]
    single_optimizer: bool
    frozen_parameters: dict[str, dict[str, str]]
    state: JsonValue


_MANIFEST = TypeAdapter(
    Annotated[
        CheckpointManifest | MultiOptimizerManifest | FrozenReferenceManifest, Field(discriminator="schema_version")
    ]
)


def _integer(value: StateValue) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Checkpoint counters must be nonnegative integers")
    return value


def _tensor(value: StateValue) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError("Expected checkpoint tensor")
    return value


def restore_checkpoint(
    path: Path,
    modules: Mapping[str, nn.Module],
    optimizer: OptimizerSet,
    identity: ResumeIdentity,
    *,
    cuda_devices: Sequence[int] = (),
) -> CheckpointCursor:
    """Restore exact module roles, optimizer groups, RNG and completed phase cursor.

    Validate version, identity, all parameter/buffer keys/shapes/dtypes, optimizer
    type and parameter roles before mutation. Load with assign=False to preserve
    optimizer parameter identity. Use a fresh session and discard it on restore
    failure; this operation is not a rollback transaction for arbitrary custom
    optimizer/module load implementations. No old checkpoint fallback exists.
    """
    roles = optimizer_roles(optimizer)
    with TensorReader(path) as reader:
        metadata = reader.metadata()
        if set(metadata) != {"unituner.checkpoint"}:
            raise ValueError("Not a native UniTuner training checkpoint")
        manifest = _MANIFEST.validate_json(metadata["unituner.checkpoint"])
        if manifest.identity != identity:
            raise ValueError("Checkpoint numerical identity does not match")
        if isinstance(manifest, CheckpointManifest):
            if not isinstance(optimizer, Optimizer):
                raise ValueError("Single-optimizer checkpoint requires a single optimizer")
            types = {"main": manifest.optimizer_type}
            signatures = {"main": manifest.optimizer_signature}
        else:
            if isinstance(manifest, FrozenReferenceManifest):
                if manifest.single_optimizer != isinstance(optimizer, Optimizer):
                    raise ValueError("Frozen-reference checkpoint optimizer role layout differs")
                if manifest.frozen_parameters.keys() != modules.keys():
                    raise ValueError("Frozen-reference checkpoint module roles differ")
            elif isinstance(optimizer, Optimizer):
                raise ValueError("Multi-optimizer checkpoint requires named optimizer roles")
            types, signatures = manifest.optimizer_types, manifest.optimizer_signatures
        expected_types = {name: type(item).__module__ + "." + type(item).__qualname__ for name, item in roles.items()}
        if types != expected_types:
            raise ValueError("Checkpoint optimizer type does not match")
        if signatures != {name: optimizer_signature(modules, item) for name, item in roles.items()}:
            raise ValueError("Checkpoint optimizer parameter roles do not match")
        tensors = {key: reader.read(key) for key in reader.keys()}
        used: set[str] = set()
        values = string_mapping(decode_state(manifest.state, tensors, used))
        if used != tensors.keys():
            raise ValueError("Checkpoint contains unreferenced tensors")
    if set(values) != {"modules", "optimizer", "run", "torch_rng", "python_rng", "numpy_rng", "cuda_rng"}:
        raise ValueError("Checkpoint state sections disagree")
    saved_modules = string_mapping(values["modules"])
    if saved_modules.keys() != modules.keys():
        raise ValueError("Checkpoint module roles do not match")
    states: dict[str, dict[str, Tensor]] = {}
    for name, module in modules.items():
        saved = {key: _tensor(value) for key, value in string_mapping(saved_modules[name]).items()}
        expected = module.state_dict()
        if isinstance(manifest, FrozenReferenceManifest):
            frozen = manifest.frozen_parameters[name]
            parameters = dict(module.named_parameters())
            for key, digest in frozen.items():
                if key not in parameters or parameters[key].requires_grad or key in saved:
                    raise ValueError("Frozen-reference checkpoint parameter ownership differs")
                if frozen_tensor_sha256(parameters[key]) != digest:
                    raise ValueError(f"Frozen-reference checkpoint base content differs for {name}.{key}")
                saved[key] = expected[key]
        if saved.keys() != expected.keys() or any(
            value.shape != expected[key].shape or value.dtype != expected[key].dtype for key, value in saved.items()
        ):
            raise ValueError(f"Checkpoint module tensor contract disagrees for {name!r}")
        states[name] = saved
    run = string_mapping(values["run"])
    if set(run) != {"updates", "microbatches", "samples"}:
        raise ValueError("Checkpoint run cursor fields disagree")
    result = CheckpointCursor(
        updates=_integer(run["updates"]), microbatches=_integer(run["microbatches"]), samples=_integer(run["samples"])
    )
    cpu_rng = _tensor(values["torch_rng"])
    cuda_rng = {key: _tensor(value) for key, value in string_mapping(values["cuda_rng"]).items()}
    if set(cuda_rng) != {str(device) for device in cuda_devices}:
        raise ValueError("Checkpoint CUDA RNG device set disagrees")
    python_rng, numpy_rng = values["python_rng"], values["numpy_rng"]
    if not isinstance(python_rng, tuple) or not isinstance(numpy_rng, tuple) or len(numpy_rng) != 5:
        raise ValueError("Checkpoint host RNG layouts disagree")
    # Validate native RNG formats on independent generators before changing globals.
    python_generator = random.Random()
    python_generator.setstate(python_rng)
    algorithm, keys, position, has_gauss, cached_gaussian = numpy_rng
    if not isinstance(algorithm, str) or not isinstance(cached_gaussian, float):
        raise ValueError("Invalid NumPy RNG scalar fields")
    numpy_state = (
        algorithm,
        _tensor(keys).numpy().astype(np.uint32),
        _integer(position),
        _integer(has_gauss),
        cached_gaussian,
    )
    numpy_generator = np.random.RandomState()
    numpy_generator.set_state(numpy_state)
    torch.Generator(device="cpu").set_state(cpu_rng)
    optimizer_states = (
        {"main": values["optimizer"]}
        if isinstance(manifest, CheckpointManifest)
        or (isinstance(manifest, FrozenReferenceManifest) and manifest.single_optimizer)
        else string_mapping(values["optimizer"])
    )
    if optimizer_states.keys() != roles.keys():
        raise ValueError("Checkpoint optimizer state roles do not match")
    # Validate every phase before load_state_dict can overwrite group policy.
    for name, item in roles.items():
        saved_groups = string_mapping(optimizer_states[name]).get("param_groups")
        if not isinstance(saved_groups, list) or len(saved_groups) != len(item.param_groups):
            raise ValueError("Checkpoint optimizer group layout does not match")
        for saved_group, group in zip(saved_groups, item.param_groups, strict=True):
            saved_group = string_mapping(saved_group)
            if any(saved_group.get(key) != group.get(key) for key in ("name", "learning_rate_scale")):
                raise ValueError("Checkpoint optimizer group name or learning-rate scale does not match")
    for name, item in roles.items():
        item.load_state_dict(string_mapping(optimizer_states[name]))
    for name, module in modules.items():
        module.load_state_dict(states[name], strict=True, assign=False)
    torch.set_rng_state(cpu_rng)
    random.setstate(python_rng)
    np.random.set_state(numpy_state)
    for device in cuda_devices:
        torch.cuda.set_rng_state(cuda_rng[str(device)], device)
    return result
