"""Build reproducible offline reports from saved tensors and recorded training metrics."""

import json
import math
import re
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from unituner.analysis.adapters import compare_deltas, groups, load_delta, summarize
from unituner.analysis.artifacts import Artifact, discover
from unituner.analysis.references import ReferenceArtifact, inspect_reference, resolve_reference
from unituner.analysis.statistics import compare, dense_spectrum, describe
from unituner.logging import operation_context
from unituner.progress import progress, status

REFERENCES = [
    {
        "title": "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions (Halko et al.)",
        "url": "https://arxiv.org/abs/0909.4061",
    },
    {"title": "LoRA: Low-Rank Adaptation of Large Language Models (2021)", "url": "https://arxiv.org/abs/2106.09685"},
    {
        "title": "Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation (2023)",
        "url": "https://arxiv.org/abs/2309.14859",
    },
    {
        "title": "PyTorch: singular values and matrix norms",
        "url": "https://docs.pytorch.org/docs/stable/generated/torch.linalg.svdvals.html",
    },
]


def canonical(name: str) -> str:
    """Normalize only explicit wrapper namespaces, never fuzzy/underscored model paths."""
    name = name.split("/", 1)[-1]
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def baseline_key(name: str, base: ReferenceArtifact, key_map: dict[str, str]) -> str | None:
    """Require one unambiguous key; users can supply explicit cross-layout mappings."""
    if name in key_map:
        target = key_map[name]
        if target not in base.tensors:
            raise ValueError(f"Key map target does not exist in baseline: {target}")
        return target
    if name in base.tensors:
        return name
    role = name.split("/", 1)[0]
    if role not in {"model", "weights"}:
        return None  # An encoder/discriminator tensor is not a denoiser weight with a similar name.
    resolved = resolve_reference(name, base)
    if resolved is not None:
        return resolved
    if not hasattr(base, "_canonical_index"):
        base._canonical_index = {}
        for key in base.tensors:
            base._canonical_index.setdefault(canonical(key), []).append(key)
    matches = [
        key for key in base._canonical_index.get(canonical(name), []) if key.split("/", 1)[0] in {"model", "weights"}
    ]
    return matches[0] if len(matches) == 1 else None


def effective_target(name: str, artifact: Artifact, context: dict) -> str:
    """Use recorded native target names when available; opaque old indices remain unmatched."""
    role, local = name.split("/", 1)
    targets = context.get("adapter_targets", []) if role == "adapter" else []
    if local.isdecimal() and int(local) < len(targets):
        return "model/" + targets[int(local)] + ".weight"
    if role == "adapter":
        return name + ".weight"  # Old opaque indices require an explicit target mapping.
    if role == "encoder":
        targets = json.loads(artifact.metadata.get("unituner.encoder.targets", "[]")) or context.get(
            "encoder_targets", []
        )
        match = re.fullmatch(r"layers\.(\d+)", local)
        if match and int(match[1]) < len(targets):
            return "encoder/" + targets[int(match[1])] + ".weight"
    targets = json.loads(artifact.metadata.get("unituner.analysis.targets", "{}"))
    return "weights/" + targets.get(local, local) + ".weight"


def read_metrics(project: Path) -> tuple[list[dict], list[str]]:
    """Read JSONL scalar events, retaining the latest resumed value at each update."""
    values, warnings = {}, []
    for path in sorted((project / "logs").glob("*.jsonl"), key=lambda item: (item.stat().st_mtime_ns, item.name)):
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                    metrics = event.get("metrics")
                    if not isinstance(metrics, dict):
                        continue
                    step = event.get("context", {}).get("update")
                    if step is None:
                        match = re.search(r"update=(\d+)", event.get("message", ""))
                        step = match[1] if match else None
                    if step is None:
                        continue
                    step = int(step)
                    values[step] = {
                        "step": step,
                        "values": {
                            k: v for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)
                        },
                    }
                except (ValueError, TypeError, AttributeError):
                    warnings.append(f"Skipped malformed log record: {path.name}:{number}")
    return [values[key] for key in sorted(values)], warnings


def analyze(
    project: Path,
    *,
    base: Path | None = None,
    scope: str = "trained",
    key_map: dict | None = None,
    svd_max_elements: int = 262144,
    factor_budget: int = 8_000_000,
    compare_path: Path | None = None,
    layers: tuple[str, ...] = (),
    top_k: int = 32,
    top_layers: int = 3,
    matrix_budget: int = 32_000_000,
    side: str = "right",
    heads: int | None = None,
    seed: int = 42,
) -> dict:
    """Compare compatible checkpoint series, keeping independent exports separate."""
    if scope not in {"trained", "all"} or side not in {"left", "right"}:
        raise ValueError("Invalid analysis scope or singular-vector side")
    if (
        min(factor_budget, top_k, top_layers, matrix_budget) < 1
        or svd_max_elements < 0
        or (heads is not None and heads < 1)
    ):
        raise ValueError("Analysis budgets/counts must be positive; svd_max_elements may be zero")
    artifacts = discover(project)
    context_root = project if project.is_dir() else project.parent
    if context_root.name in {"weights", "resume"}:
        context_root = context_root.parent
    context_path = context_root / "training-analysis.json"
    context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.exists() else {}
    if base is None and context.get("base_weights") and Path(context["base_weights"]).exists():
        base = Path(context["base_weights"])
    key_map = key_map or {}
    if not isinstance(key_map, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in key_map.items()
    ):
        raise ValueError("--key-map requires a JSON object mapping logical names to baseline logical names")
    report = {
        "schema": 1,
        "project": str(project.resolve()),
        "created": datetime.now().astimezone().isoformat(),
        "context": context,
        "scope": scope,
        "baseline": str(base.resolve()) if base else None,
        "options": {
            "svd_max_elements": svd_max_elements,
            "factor_budget": factor_budget,
            "top_k": top_k,
            "top_layers": top_layers,
            "matrix_budget": matrix_budget,
            "side": side,
            "heads": heads,
            "seed": seed,
        },
        "references": REFERENCES,
        "snapshots": [],
        "warnings": [],
        "metrics": [],
        "methodology": [
            "All values are descriptive diagnostics, not validation quality or an overfitting score.",
            "Exact finite-aware FP64 population moments; quantiles use at most 4096 evenly spaced entries per tensor (not random confidence estimates).",
            "Only final portable exports are analyzed. Periodic weights and resume states are excluded; stored buffers may be indistinguishable from weights.",
            "Portable exports expose all stored tensors; trained/all scope is equivalent for these exports.",
            "ΔW = alpha/r x B A; encoder adapters use unit scaling. LoHa uses the Hadamard product; native full-right LoKr uses unit scaling.",
            "Diffusers A/B honors PEFT alpha/rank patterns and use_rslora metadata; without alpha, vanilla PEFT uses unit scaling. DoRA is not approximated as vanilla LoRA.",
            "Stable rank = sum(s²)/max(s²); energy rank90 is the smallest k explaining 90% of squared singular values. Energy effective rank = exp(-sum(p log p)), p=s²/sum(s²).",
            "Convolutions are unfolded as [out, inxkernel]. Exact SVD is budgeted; skipped spectra are explicitly marked. Spectral charts display at most 128 singular values.",
            "Final exports cannot reconstruct parameter trajectories. Recorded loss curves are telemetry, not weight-change measurements.",
            "Charts show explicitly labeled subsets; layer diagnostics paginate at 24 rows and collection summaries at 24 files. Full numeric rows remain in CSV.",
            "Base matching uses exact names after wrapper removal, documented official/ComfyUI projection aliases and row slices, or an explicit key map; shape and coverage are reported. Layout compatibility does not verify base-model identity.",
            "Zero denominators have no defined relative change/cosine. Nonfinite entries are counted and excluded from finite moments.",
            "Packed quantization tensors are storage-only, not decoded weights. Use floating exports for logical weight analysis.",
            "Singular-vector similarity is |u_i^T v_j| (sign invariant). Repeated singular values permit basis rotations: prefer principal angles and Frobenius subspace overlap for interpretation.",
            "Subspace overlap = ||U^T V||_F²/k with equal nonzero dimension k; null directions are excluded. Large selected matrices use seeded randomized SVD, and captured energy is reported.",
            "Adapter subspace/head comparisons use the effective update ΔW versus merged W₀+ΔW when a compatible base is provided; ΔW is never inferred by subtracting the merged weights from itself. Head norm ratios and signed cosines use the merged weights as reference.",
            "Head plots require an explicit head count and contiguous output-row grouping. Fused QKV and model topology are not inferred from matrix dimensions.",
            "Nonfinite output metrics (including FP64 arithmetic overflow) are saved as null, not interpreted as zero.",
        ],
    }
    previous = {}
    with ExitStack() as stack:
        baseline = stack.enter_context(inspect_reference(base)) if base else None
        for artifact in progress(artifacts, "[analyze] artifacts"):
            predecessor = None  # Final exports are independent observations, never a time series.
            # A duplicate update is a separate snapshot, never an inferred time interval.
            duplicate_update = predecessor is not None and predecessor.step == artifact.step
            if duplicate_update:
                predecessor = None
                report["warnings"].append(f"Duplicate checkpoint update: {artifact.path.name}; temporal delta omitted")
            with ExitStack() as current_stack, operation_context(source=artifact.path, phase="weight analysis"):
                current_stack.enter_context(artifact)
                if predecessor is not None:
                    current_stack.enter_context(predecessor)
                adapter_groups = groups(artifact)
                previous_groups = groups(predecessor) if predecessor else {}
                names = set(artifact.tensors)
                if scope == "trained" and artifact.checkpoint:
                    names &= artifact.trained
                included_groups = {name: parts for name, parts in adapter_groups.items() if names & set(parts.values())}
                factor_names = {key for parts in included_groups.values() for key in parts.values()}
                snapshot = {
                    "file": str(artifact.path),
                    "label": artifact.path.name,
                    "step": artifact.step,
                    "series": artifact.series,
                    "checkpoint": artifact.checkpoint,
                    "duplicate_update": duplicate_update,
                    "omitted_frozen": artifact.omitted_frozen,
                    "stored_tensors": len(artifact.tensors),
                    "selected_tensors": len(names),
                    "previous": predecessor.path.name if predecessor else None,
                    "update_interval": artifact.step - predecessor.step if predecessor else None,
                    "rows": [],
                    "effective": [],
                    "alerts": [],
                    "excluded": [],
                    "roles": {},
                    "coverage": {
                        "base_matched": 0,
                        "base_unmatched": 0,
                        "previous_matched": 0,
                        "previous_unmatched": 0,
                    },
                }
                for name in progress(sorted(names), f"[analyze] {artifact.path.name}"):
                    dtype = artifact.dtypes[name]
                    if name in artifact.storage_only or not dtype.startswith(("F", "BF")):
                        snapshot["excluded"].append(
                            {
                                "name": name,
                                "reason": "quantized_storage" if name in artifact.storage_only else "nonfloating",
                                "dtype": dtype,
                            }
                        )
                        continue
                    value = artifact.read(name)
                    if dtype.startswith("F8"):
                        value = value.float()
                    stats = describe(value)
                    row = {
                        "name": name,
                        "role": name.split("/", 1)[0],
                        "shape": list(value.shape),
                        "dtype": dtype,
                        **stats,
                    }
                    # Raw factors have their own statistics; only ΔW has an adapter rank interpretation.
                    factor = name in factor_names
                    row["kind"] = "factor" if factor else "stored_weight_or_buffer"
                    row["spectrum"] = (
                        {"status": "see_effective_update"} if factor else dense_spectrum(value, svd_max_elements)
                    )
                    if predecessor is not None:
                        if name in predecessor.tensors and name not in predecessor.storage_only:
                            row["previous"] = compare(value, predecessor.read(name))
                            snapshot["coverage"][
                                "previous_matched"
                                if row["previous"]["status"] in ("matched", "nonfinite_pairs_excluded")
                                else "previous_unmatched"
                            ] += 1
                        else:
                            row["previous"] = {"status": "unmatched"}
                            snapshot["coverage"]["previous_unmatched"] += 1
                    if baseline is not None and not factor:
                        target = baseline_key(name, baseline, key_map)
                        if target is not None and target not in baseline.storage_only:
                            row["base"] = {"key": target, **compare(value, baseline.read(target))}
                            matched = row["base"]["status"] in ("matched", "nonfinite_pairs_excluded")
                        else:
                            row["base"] = {"status": "unmatched_or_storage_only"}
                            matched = False
                        snapshot["coverage"]["base_matched" if matched else "base_unmatched"] += 1
                    if stats["nonfinite"]:
                        snapshot["alerts"].append(
                            {
                                "severity": "error",
                                "name": name,
                                "message": f"{stats['nonfinite']} NaN/Inf entries. Inspect the first affected checkpoint, loss/gradient logs and precision before continuing.",
                            }
                        )
                    role = snapshot["roles"].setdefault(
                        row["role"], {"count": 0, "finite": 0, "nonfinite": 0, "squared_norm": 0.0}
                    )
                    for key in ("count", "finite", "nonfinite"):
                        role[key] += stats[key]
                    role["squared_norm"] += (stats["norm"] or 0) ** 2
                    snapshot["rows"].append(row)
                    del value
                for name, parts in progress(sorted(included_groups.items()), "[analyze] effective updates"):
                    try:
                        delta = load_delta(artifact, name, parts)
                        entry = {"name": name, "role": name.split("/", 1)[0], **summarize(delta, factor_budget)}
                        if predecessor is not None and name in previous_groups:
                            prior = load_delta(predecessor, name, previous_groups[name])
                            entry["previous"] = compare_deltas(delta, prior, factor_budget)
                            del prior
                        if baseline is not None:
                            target_name = effective_target(name, artifact, context)
                            target = baseline_key(key_map.get(name, target_name), baseline, key_map)
                            if name in key_map:
                                target = baseline_key(name, baseline, key_map)
                            if (
                                target is not None
                                and target not in baseline.storage_only
                                and math.prod(baseline.shapes[target]) == math.prod(delta.shape)
                                and baseline.shapes[target][0] == delta.shape[0]
                            ):
                                stats = describe(baseline.read(target))
                                norm = stats["norm"]
                                entry["base"] = {
                                    "status": "matched",
                                    "key": target,
                                    "norm": norm,
                                    "relative_delta": entry["norm"] / norm if norm and not stats["nonfinite"] else None,
                                }
                                snapshot["coverage"]["base_matched"] += 1
                            else:
                                entry["base"] = {"status": "unmatched_or_shape_mismatch"}
                                snapshot["coverage"]["base_unmatched"] += 1
                        if entry["norm"] == 0:
                            snapshot["alerts"].append(
                                {
                                    "severity": "observation",
                                    "name": name,
                                    "message": "Effective adapter update is zero. Expected at initialization; if it persists, check selected trainable layers, gradients and learning rate.",
                                }
                            )
                        relative = entry.get("previous", {}).get("relative_delta")
                        if relative is not None and relative > 1:
                            snapshot["alerts"].append(
                                {
                                    "severity": "observation",
                                    "name": name,
                                    "message": "Saved-interval change exceeds the previous update norm (ratio > 1). A small denominator or long interval can explain this; compare gradients/losses and fixed-sample validation before changing LR.",
                                }
                            )
                        snapshot["effective"].append(entry)
                        del delta
                    except (ValueError, RuntimeError, IndexError) as error:
                        snapshot["effective"].append(
                            {"name": name, "role": name.split("/", 1)[0], "status": "unavailable", "reason": str(error)}
                        )
                        snapshot["alerts"].append(
                            {
                                "severity": "warning",
                                "name": name,
                                "message": "Effective update unavailable: " + str(error),
                            }
                        )
                for role in snapshot["roles"].values():
                    role["norm"] = math.sqrt(role.pop("squared_norm"))
                    role["rms"] = role["norm"] / math.sqrt(role["finite"]) if role["finite"] else None
                report["snapshots"].append(snapshot)
                if not duplicate_update:
                    previous[artifact.series] = artifact
    if project.is_dir():
        report["metrics"], warnings = read_metrics(project)
        report["warnings"].extend(warnings)
    from unituner.analysis.subspaces import enrich_subspaces

    status("[analyze] Singular-vector alignment and subspace overlap")
    enrich_subspaces(
        report,
        artifacts,
        base=base,
        compare_path=compare_path,
        key_map=key_map,
        layers=layers,
        top_k=top_k,
        top_layers=top_layers,
        matrix_budget=matrix_budget,
        side=side,
        heads=heads,
        seed=seed,
    )
    from unituner.analysis.diagnostics import enrich_diagnostics

    enrich_diagnostics(report)
    return report


def finite_json(value):
    """Keep report JSON standards-compliant even when FP64 reductions overflow."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def publish(report: dict, directory: Path, *, show="auto") -> Path:
    """Publish final scientific figures and tables without HTML or temporary datasets."""
    from unituner.analysis.publication import publish as publish_scientific

    return publish_scientific(finite_json(report), directory, show=show)


def analyze_run(project: str | Path, *, output: str | Path | None = None, show="auto", **options) -> dict:
    """Notebook/library entry: analyze, save final products and display available figures."""
    project = Path(project)
    for key in ("base", "compare_path"):
        if options.get(key) is not None:
            options[key] = Path(options[key])
    destination = (
        Path(output)
        if output is not None
        else (
            (project if project.is_dir() else project.parent) / "analysis" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        )
    )
    report = analyze(project, **options)
    report["summary_path"] = str(publish(report, destination, show=show).resolve())
    return report


def run(args) -> int:
    """List/select a run or analyze an explicit path without changing training files."""
    project = args.project
    if project is None or args.list_runs:
        projects = [
            path
            for path in sorted(args.runs_root.glob("*"))
            if path.is_dir() and any(any((path / section).glob("*.safetensors")) for section in (".", "weights"))
        ]
        for index, path in enumerate(projects, 1):
            print(f"{index:>3}  {path}")
        if args.list_runs or not sys.stdin.isatty():
            if not args.list_runs:
                status("[analyze] Select a run: unituner analyze runs/<project>")
            return 0
        if not projects:
            raise ValueError(f"No training runs found in {args.runs_root}")
        selection = input("Run number: ").strip()
        if not selection.isdecimal() or not 1 <= int(selection) <= len(projects):
            raise ValueError("Select a run number from the displayed list")
        project = projects[int(selection) - 1]
    if args.svd_max_elements < 0 or args.factor_budget < 1:
        raise ValueError("Analysis budgets require svd-max-elements >= 0 and factor-budget > 0")
    mapping = json.loads(args.key_map.read_text(encoding="utf-8")) if args.key_map else None
    status("[analyze] CPU analysis; large base models require reading all selected weights")
    report = analyze_run(
        project,
        output=args.output,
        show=args.show,
        base=args.base,
        scope=args.scope,
        key_map=mapping,
        svd_max_elements=args.svd_max_elements,
        factor_budget=args.factor_budget,
        compare_path=args.compare_path,
        layers=tuple(args.layers),
        top_k=args.top_k,
        top_layers=args.top_layers,
        matrix_budget=args.matrix_budget,
        side=args.vector_side,
        heads=args.heads,
        seed=args.seed,
    )
    status(f"[analyze] Analysis saved: {report['summary_path']}")
    return 0
