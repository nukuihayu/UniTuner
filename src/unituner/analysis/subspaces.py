"""Measure sign-invariant singular-vector alignment and basis-invariant subspace overlap."""

import math
from contextlib import ExitStack
from fnmatch import fnmatchcase

import torch

from unituner.analysis.adapters import groups, load_delta
from unituner.analysis.references import inspect_reference


def singular_basis(matrix, *, top_k=32, side="right", seed=42):
    """Return nonzero leading directions; large matrices use a seeded randomized range finder."""
    matrix = matrix.double()
    if matrix.ndim != 2 or not min(matrix.shape) or not torch.isfinite(matrix).all():
        raise ValueError("Subspace analysis requires a nonempty finite matrix")
    k = min(top_k, min(matrix.shape))
    if min(matrix.shape) <= 256:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        v, method = vh.T, "exact SVD"
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            u, s, v = torch.svd_lowrank(matrix, q=min(k + 8, min(matrix.shape)), niter=4)
        method = "randomized SVD (oversampling=8, power iterations=4)"
    tolerance = s[0] * max(matrix.shape) * torch.finfo(matrix.dtype).eps
    k = min(k, int((s > tolerance).sum()))
    if not k:
        raise ValueError("Zero matrix has no identifiable singular subspace")
    energy = matrix.square().sum().item()
    return {
        "vectors": (u if side == "left" else v)[:, :k],
        "values": s[:k],
        "method": method,
        "energy_fraction": min(1.0, s[:k].square().sum().item() / energy),
        "total_energy": energy,
        "rank": k,
    }


def align(left, right):
    """Use the same nonzero dimension on both sides; exclude arbitrary null-space vectors."""
    k = min(left["rank"], right["rank"])
    cross = left["vectors"][:, :k].T @ right["vectors"][:, :k]
    cosines = torch.linalg.svdvals(cross).clamp(0, 1)
    return {
        "k": k,
        "absolute_cosine": cross.abs().clamp(0, 1).tolist(),
        "subspace_overlap": min(1.0, cross.square().sum().item() / k),
        "principal_angles_degrees": torch.rad2deg(torch.acos(cosines)).tolist(),
        "left_method": left["method"],
        "right_method": right["method"],
        "left_energy_fraction": min(1.0, left["values"][:k].square().sum().item() / left["total_energy"]),
        "right_energy_fraction": min(1.0, right["values"][:k].square().sum().item() / right["total_energy"]),
    }


def declared_heads(matrix, reference, heads, *, is_update=False):
    """Measure contiguous output-row groups only when the caller explicitly declares their count."""
    if heads < 1 or matrix.shape != reference.shape or matrix.shape[0] % heads:
        raise ValueError("Declared head count must divide output rows and both matrix shapes must match")
    delta = (matrix if is_update else matrix - reference).reshape(heads, -1)
    base = reference.reshape(heads, -1)
    norms = delta.norm(dim=1)
    denominators = base.norm(dim=1)
    products = norms * denominators
    dot = (delta * base).sum(dim=1)
    return {
        "cosine_with_reference": [
            max(-1.0, min(1.0, a.item() / b.item())) if b else None for a, b in zip(dot, products, strict=True)
        ],
        "rms": delta.square().mean(dim=1).sqrt().tolist(),
        "relative_change": [a.item() / b.item() if b else None for a, b in zip(norms, denominators, strict=True)],
        "interpretation": "User-declared contiguous output-row groups; fused QKV layouts are not inferred",
    }


def enrich_subspaces(
    report,
    artifacts,
    *,
    base=None,
    compare_path=None,
    key_map=None,
    layers=(),
    top_k=32,
    top_layers=3,
    matrix_budget=32_000_000,
    side="right",
    heads=None,
    seed=42,
):
    """Analyze selected latest snapshots; never fabricate unavailable baseline or topology mappings."""
    from unituner.analysis.run import baseline_key, effective_target

    key_map = key_map or {}
    report["subspaces"] = []
    latest = {}
    for index, snapshot in enumerate(report["snapshots"]):
        if snapshot.get("duplicate_update"):
            continue
        key = snapshot["file"]
        latest[key] = index
    with ExitStack() as stack:
        baseline = stack.enter_context(inspect_reference(base)) if base else None
        comparison = stack.enter_context(inspect_reference(compare_path)) if compare_path else None
        for index in latest.values():
            snapshot, artifact = report["snapshots"][index], artifacts[index]
            candidates = [(row, True) for row in snapshot["effective"] if "norm" in row]
            candidates += [
                (row, False) for row in snapshot["rows"] if row["kind"] != "factor" and len(row["shape"]) >= 2
            ]
            candidates = sorted(candidates, key=lambda item: item[0].get("rms") or 0, reverse=True)
            if layers:
                candidates = [
                    (row, adapter)
                    for row, adapter in candidates
                    if any(fnmatchcase(row["name"], pattern) for pattern in layers)
                ]
            else:
                oversized = sum(math.prod(row["shape"]) > matrix_budget for row, _ in candidates)
                candidates = [
                    (row, adapter) for row, adapter in candidates if math.prod(row["shape"]) <= matrix_budget
                ][:top_layers]
                if oversized:
                    report["warnings"].append(
                        f"{snapshot['label']}: {oversized} matrices exceed the subspace budget; automatic selection uses eligible layers"
                    )
            with ExitStack() as current_stack:
                current_stack.enter_context(artifact)
                grouped = groups(artifact)
                prior = None
                if snapshot["previous"]:
                    prior = next(
                        (
                            item
                            for item in artifacts
                            if item.path.name == snapshot["previous"] and item.series == artifact.series
                        ),
                        None,
                    )
                    if prior:
                        current_stack.enter_context(prior)
                for row, adapter in candidates:
                    item = {
                        "snapshot": snapshot["label"],
                        "name": row["name"],
                        "side": side,
                        "pairs": [],
                        "heads": None,
                    }
                    try:
                        shape = row["shape"]
                        if math.prod(shape) > matrix_budget:
                            raise ValueError(
                                f"Matrix exceeds subspace budget ({math.prod(shape):,} > {matrix_budget:,} elements)"
                            )
                        name = row["name"]
                        target = effective_target(name, artifact, report["context"]) if adapter else name
                        base_key = (
                            baseline_key(name if name in key_map else target, baseline, key_map) if baseline else None
                        )
                        if adapter:
                            delta = load_delta(artifact, name, grouped[name])
                            current = delta.rows(0, delta.shape[0])
                            update_label = "LoRA ΔW" if delta.kind in {"lora", "encoder-lora"} else "Adapter ΔW"
                            del delta
                        else:
                            value = artifact.read(name)
                            current = value.double().reshape(value.shape[0], -1)
                        reference = None
                        merged_reference = False
                        reference_label, current_label = "", ""
                        if base_key is not None and base_key not in baseline.storage_only:
                            value = baseline.read(base_key)
                            reference = value.double().reshape(value.shape[0], -1)
                            reference_label = "Base W₀"
                            if current.shape != reference.shape:
                                raise ValueError("Baseline and current matrix shapes differ")
                            if adapter:
                                reference = reference + current
                                merged_reference = True
                                reference_label, current_label = "Merged W₀+ΔW", update_label
                            else:
                                current_label = "Trained weights"
                        elif prior is not None:
                            if adapter and name in groups(prior):
                                delta = load_delta(prior, name, groups(prior)[name])
                                reference = delta.rows(0, delta.shape[0])
                                del delta
                                reference_label, current_label = "Previous adapter update", "Current adapter update"
                            elif not adapter and name in prior.tensors:
                                value = prior.read(name)
                                reference = value.double().reshape(value.shape[0], -1)
                                reference_label, current_label = "Previous weights", "Current weights"
                        if reference is None:
                            raise ValueError(
                                "No compatible baseline or previous saved matrix; use --base and/or --key-map"
                            )
                        if current.shape != reference.shape:
                            raise ValueError("Comparison matrix shapes differ")
                        left = singular_basis(reference, top_k=top_k, side=side, seed=seed)
                        right = singular_basis(current, top_k=top_k, side=side, seed=seed)
                        change = current if merged_reference else current - reference
                        denominator = change.square().sum().item()
                        projection = change @ left["vectors"] if side == "right" else left["vectors"].T @ change
                        item["change_energy_in_reference_subspace"] = (
                            min(1.0, projection.square().sum().item() / denominator) if denominator else None
                        )
                        item["pairs"].append({"left": reference_label, "right": current_label, **align(left, right)})
                        item["reference_label"] = reference_label
                        item["update_label"] = current_label if adapter else "Weight change"
                        if comparison is not None:
                            other_key = baseline_key(target, comparison, key_map)
                            if other_key is not None and other_key not in comparison.storage_only:
                                value = comparison.read(other_key)
                                other = value.double().reshape(value.shape[0], -1)
                                if other.shape == current.shape:
                                    third = singular_basis(other, top_k=top_k, side=side, seed=seed)
                                    label = "Comparison weights"
                                    item["pairs"].extend(
                                        [
                                            {"left": reference_label, "right": label, **align(left, third)},
                                            {"left": label, "right": current_label, **align(third, right)},
                                        ]
                                    )
                                else:
                                    item["comparison_note"] = "Comparison shape mismatch"
                            else:
                                item["comparison_note"] = "Comparison key unavailable or quantized storage"
                        if heads is not None:
                            item["heads"] = declared_heads(current, reference, heads, is_update=merged_reference)
                            item["heads"]["comparison"] = f"{current_label} vs {reference_label}"
                        item["status"] = "available"
                    except (ValueError, RuntimeError, IndexError) as error:
                        item["status"], item["reason"] = "unavailable", str(error)
                    report["subspaces"].append(item)
            if layers and not candidates:
                report["warnings"].append(f"No layer selectors matched {snapshot['label']}")
