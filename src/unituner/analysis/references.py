"""Expose exact floating projection views across official and ComfyUI weight layouts."""

from collections import defaultdict
from dataclasses import dataclass, field

import torch

from unituner.analysis.artifacts import Artifact, inspect_artifact
from unituner.artifacts.formats.precision_policy import resolve_family, unprefixed
from unituner.artifacts.formats.transformer import official_projection_sources


@dataclass
class ReferenceArtifact(Artifact):
    """Own a reference-only projection index; exported artifact inventories stay untouched."""

    views: dict[str, tuple[tuple[str, int, int], ...]] = field(default_factory=dict)

    def read(self, name: str) -> torch.Tensor:
        """Read a mapped projection, allocating at most one joined reference matrix."""
        if name in self.views:
            pieces = [self.read(key)[start:stop] for key, start, stop in self.views[name]]
            return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        return super().read(name)


def inspect_reference(path):
    """Keep virtual aliases confined to base/compare readers, never analyzed weights."""
    artifact = inspect_artifact(path)
    artifact.storage_only.update(name for name, dtype in artifact.dtypes.items() if not dtype.startswith(("F", "BF")))
    return ReferenceArtifact(**vars(artifact))


def reference_views(artifact):
    """Index aliases and row slices using headers; never load or dequantize a complete model.

    Virtual references retain original storage-only flags. Packed quantization
    is deliberately unavailable for numeric comparison, rather than interpreting
    integer bytes as logical weights. Conflicting namespaces stay ambiguous.
    """
    if getattr(artifact, "_reference_views_installed", False):
        return
    artifact._reference_views_installed = True
    inventory = defaultdict(list)
    for name in artifact.tensors:
        role, local = name.split("/", 1)
        if role in {"weights", "model"}:
            inventory[unprefixed(local)].append(name)
    originals = {local: names[0] for local, names in inventory.items() if len(names) == 1}

    def add(local, pieces):
        name = "weights/" + local
        if name in artifact.tensors or len(inventory.get(local, ())) > 1:
            return
        shapes = [artifact.shapes[key] for key, _, _ in pieces]
        if not shapes or any(not shape or shape[1:] != shapes[0][1:] for shape in shapes):
            return
        if len({artifact.dtypes[key] for key, _, _ in pieces}) != 1:
            return
        if any(
            start < 0 or stop <= start or stop > shape[0]
            for (_, start, stop), shape in zip(pieces, shapes, strict=True)
        ):
            return
        artifact.tensors[name] = artifact.tensors[pieces[0][0]]
        artifact.shapes[name] = (sum(stop - start for _, start, stop in pieces), *shapes[0][1:])
        artifact.dtypes[name] = artifact.dtypes[pieces[0][0]]
        artifact.views[name] = tuple(pieces)
        if any(key in artifact.storage_only for key, _, _ in pieces):
            artifact.storage_only.add(name)

    for local, name in list(originals.items()):
        shape = artifact.shapes[name]
        if shape:
            add(local, [(name, 0, shape[0])])

    try:
        family = resolve_family("auto", artifact.metadata, originals)
    except ValueError as error:
        if "Ambiguous model layout" not in str(error):
            raise
        family = "generic"
    # These fused paths identify their row semantics even in partial artifacts.
    for local, name in list(originals.items()):
        shape = artifact.shapes[name]
        if len(shape) != 2 or not local.endswith(".weight"):
            continue
        rows, width = shape
        parts = []
        if local.endswith(".img_mlp.gate_up.weight") and rows % 2 == 0:
            half = rows // 2
            parts = [
                (local.replace("gate_up", field), i * half, (i + 1) * half)
                for i, field in enumerate(("gate_layer", "proj"))
            ]
        elif ".attention.qkv." in local:
            output = originals.get(local.replace(".qkv.", ".out.")) or originals.get(
                local.replace(".qkv.", ".to_out.0.")
            )
            query = artifact.shapes[output][1] if output and len(artifact.shapes[output]) == 2 else width
            if rows > query and (rows - query) % 2 == 0:
                kv = (rows - query) // 2
                parts = [
                    (local.replace("qkv", "to_q"), 0, query),
                    (local.replace("qkv", "to_k"), query, query + kv),
                    (local.replace("qkv", "to_v"), query + kv, rows),
                ]
        elif local.endswith((".img_attn.qkv.weight", ".txt_attn.qkv.weight")) and rows == 3 * width:
            stream = "img" if ".img_attn." in local else "txt"
            prefix = local.split(f".{stream}_attn.")[0].replace("double_blocks.", "transformer_blocks.")
            fields = ("to_q", "to_k", "to_v") if stream == "img" else ("add_q_proj", "add_k_proj", "add_v_proj")
            parts = [(f"{prefix}.attn.{field}.weight", i * width, (i + 1) * width) for i, field in enumerate(fields)]
        elif local.startswith("single_blocks.") and local.endswith(".linear1.weight") and rows > 3 * width:
            prefix = local.removesuffix("linear1.weight").replace("single_blocks.", "single_transformer_blocks.")
            parts = [
                (prefix + f"attn.to_{field}.weight", i * width, (i + 1) * width)
                for i, field in enumerate(("q", "k", "v"))
            ]
            parts.append((prefix + "proj_mlp.weight", 3 * width, rows))
            add(prefix + "attn.to_qkv_mlp_proj.weight", [(name, 0, rows)])
        elif local.endswith(".attn.qkv_proj.weight") and rows % 3 == 0:
            third = rows // 3
            parts = [
                (
                    local.replace("blocks.", "transformer_blocks.", 1).replace("qkv_proj", f"to_{field}"),
                    i * third,
                    (i + 1) * third,
                )
                for i, field in enumerate(("q", "k", "v"))
            ]
        elif family == "flux3" and local.endswith(".qkv.weight") and rows % 3 == 0:
            third = rows // 3
            parts = [
                (local.replace(".qkv.", f".{field}_proj."), i * third, (i + 1) * third)
                for i, field in enumerate(("q", "k", "v"))
            ]
        for target, start, stop in parts:
            add(target, [(name, start, stop)])
            alias = "weights/" + target
            if alias in artifact.tensors:
                originals.setdefault(target, alias)
        if local.startswith("single_blocks.") and local.endswith(".linear2.weight"):
            add(
                local.replace("single_blocks.", "single_transformer_blocks.").replace(".linear2.", ".attn.to_out."),
                [(name, 0, rows)],
            )

    # Native-to-official one-to-one aliases, including H3's SwiGLU row order.
    for local, name in originals.items():
        shape = artifact.shapes[name]
        if not shape:
            continue
        targets = official_projection_sources(family, local)
        if len(targets) == 1 and targets[0] != local:
            if family == "minimax-h3" and local.endswith(".mlp.fc1.weight"):
                if shape[0] % 2:
                    continue
                half = shape[0] // 2
                add(targets[0], [(name, half, shape[0]), (name, 0, half)])
            else:
                add(targets[0], [(name, 0, shape[0])])
        if family == "z-image":
            target = local.replace(".attention.out.", ".attention.to_out.0.")
            target = target.replace(".attention.q_norm.", ".attention.norm_q.")
            target = target.replace(".attention.k_norm.", ".attention.norm_k.")
            if target != local:
                add(target, [(name, 0, shape[0])])

    artifact.reference_family = family


def resolve_reference(name, artifact):
    """Resolve native projections from split official tensors when no direct alias exists."""
    reference_views(artifact)
    local = unprefixed(name.split("/", 1)[-1])
    direct = "weights/" + local
    if direct in artifact.tensors:
        return direct
    family = artifact.reference_family
    sources = official_projection_sources(family, local)
    if local.endswith(".img_mlp.gate_up.weight"):
        sources = tuple(local.replace("gate_up", field) for field in ("gate_layer", "proj"))
    elif ".attention.qkv." in local:
        sources = tuple(local.replace(".qkv.", f".to_{field}.") for field in ("q", "k", "v"))
    elif family == "flux3" and local.endswith(".qkv.weight"):
        sources = tuple(local.replace(".qkv.", f".{field}_proj.") for field in ("q", "k", "v"))
    elif family == "flux" and local.startswith("single_transformer_blocks."):
        if local.endswith(".attn.to_qkv_mlp_proj.weight"):
            prefix = local.removesuffix("attn.to_qkv_mlp_proj.weight")
            sources = tuple(prefix + field + ".weight" for field in ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"))
        elif local.endswith(".attn.to_out.weight"):
            sources = (local.replace(".attn.to_out.", ".proj_out."),)
    elif family == "flux2":
        sources = tuple(
            key.replace(".ff.net.0.proj.", ".ff.linear_in.")
            .replace(".ff.net.2.", ".ff.linear_out.")
            .replace(".ff_context.net.0.proj.", ".ff_context.linear_in.")
            .replace(".ff_context.net.2.", ".ff_context.linear_out.")
            for key in sources
        )
    elif family == "z-image":
        sources = tuple(key.replace(".attention.out.", ".attention.to_out.0.") for key in sources)
    if sources == (local,):
        return None
    keys = ["weights/" + key for key in sources]
    if any(key not in artifact.tensors for key in keys):
        return None
    shapes = [artifact.shapes[key] for key in keys]
    if any(not shape or shape[1:] != shapes[0][1:] for shape in shapes):
        return None
    if len({artifact.dtypes[key] for key in keys}) != 1:
        return None
    pieces = [(key, 0, shape[0]) for key, shape in zip(keys, shapes, strict=True)]
    if family == "minimax-h3" and local.endswith(".mlp.fc1.weight"):
        if shapes[0][0] % 2:
            return None
        half = shapes[0][0] // 2
        pieces = [(keys[0], half, shapes[0][0]), (keys[0], 0, half)]
    artifact.tensors[direct] = artifact.tensors[keys[0]]
    artifact.shapes[direct] = (sum(stop - start for _, start, stop in pieces), *shapes[0][1:])
    artifact.dtypes[direct] = artifact.dtypes[keys[0]]
    artifact.views[direct] = tuple(pieces)
    if any(key in artifact.storage_only for key in keys):
        artifact.storage_only.add(direct)
    return direct
