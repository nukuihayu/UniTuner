"""Expose logical module tensors from native checkpoints and weight artifacts."""

import hashlib
import json
import re
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import torch

from unitunner.artifacts.tensors import TensorReader, resolve_shards


def mapping(node: dict) -> dict:
    """Read a tagged mapping without loading any tensor or executing serialized code."""
    if not isinstance(node, dict) or node.get("kind") != "mapping":
        raise ValueError("Analysis expected a native checkpoint mapping")
    result = {}
    for entry in node["entries"]:
        key = entry["key"]
        if not isinstance(key, (int, str)) or key in result:
            raise ValueError("Duplicate or invalid checkpoint mapping key")
        result[key] = entry["value"]
    return result


@dataclass
class Artifact:
    """Retain headers and logical names, excluding optimizer and RNG tensors."""

    path: Path
    tensors: dict[str, tuple[Path, str]] = field(default_factory=dict)
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    dtypes: dict[str, str] = field(default_factory=dict)
    trained: set[str] = field(default_factory=set)
    metadata: dict[str, str] = field(default_factory=dict)
    step: int | None = None
    series: str = "artifact"
    checkpoint: bool = False
    omitted_frozen: int = 0
    storage_only: set[str] = field(default_factory=set)
    _readers: dict = field(default_factory=dict)
    _stack: ExitStack | None = None

    def __enter__(self):
        self._stack = ExitStack()
        try:
            for path, _ in self.tensors.values():
                if path not in self._readers:
                    self._readers[path] = self._stack.enter_context(TensorReader(path))
        except Exception:
            self._stack.close()
            self._readers.clear()
            raise
        return self

    def __exit__(self, *args):
        if self._stack is not None:
            self._stack.close()
        self._readers.clear()

    def read(self, name: str) -> torch.Tensor:
        """Read one tensor from an open artifact, preserving original dtype."""
        path, key = self.tensors[name]
        return self._readers[path].read(key)


def inspect_artifact(path: Path) -> Artifact:
    """Index safetensors or indexed shards; distinguish native module roles explicitly."""
    artifact = Artifact(path.resolve())
    for shard in resolve_shards(path):
        with TensorReader(shard) as reader:
            metadata = reader.metadata()
            artifact.metadata.update(metadata)
            keys = reader.keys()
            if "unitunner.checkpoint" in metadata:
                if artifact.tensors:
                    raise ValueError("Native training checkpoints must be single files")
                manifest = json.loads(metadata["unitunner.checkpoint"])
                if manifest.get("schema_version") not in (1, 2, 3):
                    raise ValueError("Unsupported training checkpoint schema")
                artifact.checkpoint = True
                state = mapping(manifest["state"])
                step = mapping(state["run"])["updates"]
                if step.get("kind") != "scalar" or type(step.get("value")) is not int or step["value"] < 0:
                    raise ValueError("Invalid checkpoint update counter")
                artifact.step = step["value"]
                artifact.series = (
                    "checkpoint/"
                    + hashlib.sha256(json.dumps(manifest["identity"], sort_keys=True).encode()).hexdigest()[:16]
                )
                signatures = manifest.get("optimizer_signatures", {"main": manifest.get("optimizer_signature", [])})
                artifact.trained = {
                    value.replace(".", "/", 1) for groups in signatures.values() for group in groups for value in group
                }
                artifact.omitted_frozen = sum(len(values) for values in manifest.get("frozen_parameters", {}).values())
                selected = {}
                for role, values in mapping(state["modules"]).items():
                    for name, node in mapping(values).items():
                        if node.get("kind") != "tensor" or node.get("key") not in keys:
                            raise ValueError(f"Invalid checkpoint module tensor: {role}/{name}")
                        selected[f"{role}/{name}"] = node["key"]
            else:
                role = "encoder" if "unitunner.encoder.targets" in metadata else "weights"
                selected = {f"{role}/{name}": name for name in keys}
                if "unitunner.analysis.step" in metadata and "unitunner.analysis.identity" in metadata:
                    artifact.step = int(metadata["unitunner.analysis.step"])
                    if artifact.step < 0:
                        raise ValueError("Export update must be nonnegative")
                    artifact.series = (
                        "export/"
                        + role
                        + "/"
                        + hashlib.sha256(metadata["unitunner.analysis.identity"].encode()).hexdigest()[:16]
                    )
            quant_prefixes = [name.removesuffix("comfy_quant") for name in keys if name.endswith("comfy_quant")]
            for name, key in selected.items():
                if name in artifact.tensors:
                    raise ValueError(f"Duplicate logical tensor across shards: {name}")
                artifact.tensors[name] = shard, key
                artifact.shapes[name] = reader.shape(key)
                artifact.dtypes[name] = reader.dtype(key)
                if any(key.startswith(prefix) for prefix in quant_prefixes):
                    artifact.storage_only.add(name)
    if not artifact.tensors:
        raise ValueError(f"No model/module tensors found: {path}")
    return artifact


def is_periodic(path: Path) -> bool:
    """Recognize UniTunner's reserved periodic-export filename suffix."""
    return bool(re.search(r"(?:\.state-\d+|-\d{8,}(?:\.encoder)?)\.safetensors(?:\.index\.json)?$", path.name))


def discover(project: Path) -> list[Artifact]:
    """Read final portable exports only; never fall back to resume or periodic files."""
    if project.is_file():
        if is_periodic(project) or project.parent.name == "resume":
            raise ValueError("Select final exported weights, not periodic weights or resume state")
        artifact = inspect_artifact(project)
        if artifact.checkpoint:
            raise ValueError("Training snapshots are not final weights; select the weights/ export")
        return [artifact]
    if not project.is_dir():
        raise FileNotFoundError(project)
    if project.name == "resume":
        raise ValueError("resume/ contains training state; select final weights in weights/")
    directory = project / "weights" if (project / "weights").is_dir() else project
    paths = [
        path
        for pattern in ("*.safetensors.index.json", "*.safetensors")
        for path in sorted(directory.glob(pattern))
        if not is_periodic(path)
    ]
    # Reserve shards from *all* indexes, including excluded periodic exports.
    indexed_shards = {shard for index in directory.glob("*.safetensors.index.json") for shard in resolve_shards(index)}
    paths = [path for path in paths if path not in indexed_shards]
    used, artifacts = set(), []
    for path in paths:
        shards = set(resolve_shards(path))
        if shards <= used:
            continue
        if shards & used:
            raise ValueError(f"Overlapping weight artifacts: {path}")
        artifact = inspect_artifact(path)
        if not artifact.checkpoint:
            artifacts.append(artifact)
        used.update(shards)
    if not artifacts:
        if (project / "transformer").is_dir():
            return [inspect_artifact(project)]
        raise ValueError(f"No final exported weights in {directory}; periodic weights and resume states are excluded")
    return sorted(artifacts, key=lambda item: str(item.path))
