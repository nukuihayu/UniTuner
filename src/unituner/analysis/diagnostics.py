"""Derived descriptive diagnostics with explicit comparison populations and denominators."""

import math
import re
import statistics
from collections import defaultdict


def enrich_diagnostics(report):
    """Measure allocation, LoRA rank usage and robust within-peer scale deviations."""
    for snapshot in report["snapshots"]:
        effective = bool(snapshot["effective"])
        source = snapshot["effective"] if effective else [r for r in snapshot["rows"] if r["kind"] != "factor"]
        has_base = not effective and any(r.get("base", {}).get("delta_norm") is not None for r in source)
        rows, peers = [], defaultdict(list)
        for row in sorted(
            source, key=lambda r: [int(p) if p.isdecimal() else p for p in re.split(r"(\d+)", r["name"])]
        ):
            norm = row.get("norm") if effective or not has_base else row.get("base", {}).get("delta_norm")
            usable = norm is not None and math.isfinite(norm) and not row.get("nonfinite")
            shape = row.get("shape", [])
            rms_kind = "effective_update" if effective else "base_difference" if has_base else "stored_weight"
            if has_base:
                usable = usable and not row.get("base", {}).get("excluded_pairs")
                count = math.prod(shape)
                rms = norm / math.sqrt(count) if usable and count else None
            else:
                rms = row.get("rms")
            capacity = (
                min(row["rank"], *shape)
                if row.get("algorithm") in {"lora", "encoder-lora"} and row.get("rank") and shape
                else None
            )
            spectrum = row.get("spectrum", {})
            item = {
                "name": row["name"],
                "role": row["role"],
                "norm": norm if usable else None,
                "rms": rms,
                "rms_kind": rms_kind,
                "relative_delta": row.get("base", {}).get("relative_delta"),
                "lora_rank_capacity": capacity,
                "stable_rank_fraction": spectrum.get("stable_rank", 0) / capacity
                if capacity and "stable_rank" in spectrum
                else None,
                "rank90_fraction": spectrum.get("energy_rank90", 0) / capacity
                if capacity and "energy_rank90" in spectrum
                else None,
                "energy_effective_rank": spectrum.get("energy_effective_rank"),
                "top_singular_energy_fraction": spectrum.get("top_energy_fraction"),
                "peer_log_rms_score": None,
                "peer_count": 0,
            }
            rows.append(item)
            # Compare only matching role, algorithm, matrix geometry and final module token.
            peer = (
                row["role"],
                row.get("algorithm", "dense"),
                tuple(shape),
                row["name"].removesuffix(".weight").split(".")[-1],
            )
            if item["rms"] and math.isfinite(item["rms"]):
                peers[peer].append(item)
        for population in peers.values():
            for item in population:
                item["peer_count"] = len(population)
            if len(population) < 5:
                continue
            logs = [math.log10(item["rms"]) for item in population]
            median = statistics.median(logs)
            mad = statistics.median(abs(v - median) for v in logs)
            if mad == 0:
                continue
            for item, value in zip(population, logs, strict=True):
                item["peer_log_rms_score"] = 0.67448975 * (value - median) / mad
                if abs(item["peer_log_rms_score"]) > 3.5:
                    snapshot["alerts"].append(
                        {
                            "severity": "observation",
                            "name": item["name"],
                            "message": f"Log-RMS differs from {len(population)} same-role/shape/module peers (robust score {item['peer_log_rms_score']:.2f}). This is a heuristic, not a significance test; check layer purpose, baseline norm and validation before changing optimization.",
                        }
                    )
        roles = []
        for role in sorted({r["role"] for r in rows}):
            population = [r for r in rows if r["role"] == role and r["norm"] is not None]
            maximum = max((r["norm"] for r in population), default=0)
            energy = [(r["norm"] / maximum) ** 2 if maximum else 0 for r in population]
            total = sum(energy)
            for item, value in zip(population, energy, strict=True):
                item["energy_fraction"] = value / total if total else None
            fractions = sorted((v / total for v in energy), reverse=True) if total else []
            cumulative, running = [], 0
            for value in fractions:
                running += value
                cumulative.append(running)
            roles.append(
                {
                    "role": role,
                    "covered_layers": len(population),
                    "total_layers": sum(r["role"] == role for r in rows),
                    "energy_kind": "effective_update"
                    if effective
                    else "base_difference"
                    if has_base
                    else "stored_weight",
                    "top10_energy_fraction": sum(fractions[:10]) if fractions else None,
                    "effective_layer_count": 1 / sum(v * v for v in fractions) if fractions else None,
                    "layers_for_90pct_energy": next((i for i, v in enumerate(cumulative, 1) if v >= 0.9), None),
                    "cumulative_energy": cumulative,
                }
            )
        snapshot["diagnostics"] = {"layers": rows, "roles": roles}
    report["methodology"] += [
        "Layer energy is squared Frobenius norm of effective updates, matched base differences, or (without a base) stored weights; roles are analyzed separately. Large tensors naturally contribute more energy; inspect RMS alongside energy.",
        "Effective layer count = 1/sum(p_layer²); 90% concentration counts the largest-energy layers. This describes allocation, not optimal allocation.",
        "LoRA rank usage divides stable rank and 90%-energy rank by min(configured rank, output dimension, input dimension). It is not a direct estimate of required training rank. LoHa/LoKr do not use this LoRA-specific denominator.",
        "Diagnostic RMS uses effective ΔW for adapters and matched W_trained-W₀ for full-weight artifacts with a base; stored RMS is used only without a base. Per-layer raw stored moments remain separately available in layers.csv. Unmatched/nonfinite differences are excluded from update RMS. The rms_kind column identifies the source.",
        "Peer score = 0.67448975*(log10(RMS)-median)/MAD within matching role, algorithm, shape and final module token. At least five positive finite observations and nonzero MAD are required. |score|>3.5 is a heuristic flag, not a calibrated significance or failure threshold.",
    ]
