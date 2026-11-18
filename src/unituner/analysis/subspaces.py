"""Measure sign-invariant singular-vector alignment and basis-invariant subspace overlap."""

import math
from contextlib import ExitStack
from fnmatch import fnmatchcase

import torch

from unituner.analysis.adapters import groups, load_delta
from unituner.analysis.low_rank import FactorMatrix, product_basis
from unituner.analysis.references import inspect_reference


def singular_basis(matrix, *, top_k=32, side="right", seed=42):
    """Return nonzero leading directions; large matrices use a seeded randomized range finder."""
    if isinstance(matrix, FactorMatrix):
        return product_basis(matrix.left, matrix.right, top_k=top_k, side=side)
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
    if isinstance(matrix, FactorMatrix) and isinstance(reference, FactorMatrix):
        change = matrix if is_update else matrix.subtract(reference)
        width = matrix.shape[0] // heads
        norms, denominators, dots = [], [], []
        for index in range(heads):
            start, stop = index * width, (index + 1) * width
            a = FactorMatrix(change.left[start:stop], change.right)
            b = FactorMatrix(reference.left[start:stop], reference.right)
            norms.append(a.norm())
            denominators.append(b.norm())
            dots.append(((a.left.T @ b.left) * (a.right @ b.right.T)).sum().item())
        return {
            "cosine_with_reference": [
                max(-1.0, min(1.0, dot / (a * b))) if a and b else None
                for dot, a, b in zip(dots, norms, denominators, strict=True)
            ],
            "rms": [value / math.sqrt(width * matrix.shape[1]) for value in norms],
            "relative_change": [a / b if b else None for a, b in zip(norms, denominators, strict=True)],
            "interpretation": "User-declared contiguous output-row groups; fused QKV layouts are not inferred",
        }
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

                def work_size(row, adapter):
                    if adapter and row.get("algorithm") in {"lora", "encoder-lora"}:
                        rank = row["rank"]
                        return max(sum(row["shape"]) * rank, rank * rank)
                    return math.prod(row["shape"])

                oversized = sum(work_size(row, adapter) > matrix_budget for row, adapter in candidates)
                candidates = [
                    (row, adapter) for row, adapter in candidates if work_size(row, adapter) <= matrix_budget
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
                        if not adapter and math.prod(shape) > matrix_budget:
                            raise ValueError(
                                f"Matrix exceeds subspace budget ({math.prod(shape):,} > {matrix_budget:,} elements)"
                            )
                        name = row["name"]
                        target = effective_target(name, artifact, report["context"]) if adapter else name
                        base_key = (
                            baseline_key(name if name in key_map else target, baseline, key_map) if baseline else None
                        )
                        current_factors = None
                        if adapter:
                            delta = load_delta(artifact, name, grouped[name])
                            factors = delta.low_rank(matrix_budget)
                            if (
                                factors is not None
                                and sum(t.numel() for t in factors) <= matrix_budget
                                and factors[0].shape[1] ** 2 <= matrix_budget
                            ):
                                current = FactorMatrix(*factors)
                                current_factors = current
                            elif math.prod(shape) <= matrix_budget:
                                current = delta.rows(0, delta.shape[0])
                            else:
                                raise ValueError("Adapter factors exceed the subspace budget")
                            update_label = "LoRA ΔW" if delta.kind in {"lora", "encoder-lora"} else "Adapter ΔW"
                            del delta
                        else:
                            value = artifact.read(name)
                            current = value.double().reshape(value.shape[0], -1)
                        reference = None
                        merged_reference = False
                        reference_label, current_label = "", ""
                        if base_key is not None and base_key not in baseline.storage_only:
                            if math.prod(baseline.shapes[base_key]) > matrix_budget:
                                raise ValueError("Baseline matrix exceeds the subspace budget")
                            if isinstance(current, FactorMatrix):
                                if math.prod(current.shape) > matrix_budget:
                                    raise ValueError("Merged baseline matrix exceeds the subspace budget")
                                current = current.rows(0, current.shape[0])
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
                                factors = delta.low_rank(matrix_budget)
                                if (
                                    isinstance(current, FactorMatrix)
                                    and factors is not None
                                    and sum(t.numel() for t in factors) <= matrix_budget
                                    and factors[0].shape[1] ** 2 <= matrix_budget
                                ):
                                    reference = FactorMatrix(*factors)
                                elif max(math.prod(current.shape), math.prod(delta.shape)) <= matrix_budget:
                                    if isinstance(current, FactorMatrix):
                                        current = current.rows(0, current.shape[0])
                                    reference = delta.rows(0, delta.shape[0])
                                else:
                                    raise ValueError("Previous adapter factors exceed the subspace budget")
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
                        right = singular_basis(
                            current_factors if current_factors is not None else current,
                            top_k=top_k,
                            side=side,
                            seed=seed,
                        )
                        if isinstance(current, FactorMatrix):
                            change = current.subtract(reference)
                            if (
                                max(change.left.numel() + change.right.numel(), change.left.shape[1] ** 2)
                                > matrix_budget
                            ):
                                raise ValueError("Checkpoint difference core exceeds the subspace budget")
                            denominator = change.norm() ** 2
                            projected = change.projected_energy(left["vectors"], side)
                        else:
                            change = current if merged_reference else current - reference
                            denominator = change.square().sum().item()
                            projection = change @ left["vectors"] if side == "right" else left["vectors"].T @ change
                            projected = projection.square().sum().item()
                        item["change_energy_in_reference_subspace"] = (
                            min(1.0, projected / denominator) if denominator else None
                        )
                        item["pairs"].append({"left": reference_label, "right": current_label, **align(left, right)})
                        item["reference_label"] = reference_label
                        item["update_label"] = current_label if adapter else "Weight change"
                        if comparison is not None:
                            other_key = baseline_key(target, comparison, key_map)
                            if other_key is not None and other_key not in comparison.storage_only:
                                if math.prod(comparison.shapes[other_key]) > matrix_budget:
                                    raise ValueError("Comparison matrix exceeds the subspace budget")
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
