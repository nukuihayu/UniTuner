"""Analyze actual adapter updates, respecting native alpha and encoder scaling."""

import math
from dataclasses import dataclass

import torch

from unitunner.analysis.artifacts import Artifact
from unitunner.analysis.statistics import CHUNK, spectrum

SUFFIXES = {
    "lora_down.weight": "down",
    "lora_up.weight": "up",
    "hada_w1_a": "a1",
    "hada_w1_b": "b1",
    "hada_w2_a": "a2",
    "hada_w2_b": "b2",
    "lokr_w1": "left",
    "lokr_w2": "right",
    "lokr_w2_a": "right_up",
    "lokr_w2_b": "right_down",
    "down": "down",
    "up": "up",
    "a1": "a1",
    "b1": "b1",
    "a2": "a2",
    "b2": "b2",
    "left": "left",
    "right": "right",
    "right_up": "right_up",
    "right_down": "right_down",
    "alpha": "alpha",
}


def groups(artifact: Artifact) -> dict[str, dict[str, str]]:
    """Recognize complete native/exported factor sets; do not split underscored layer names."""
    candidates = {}
    for key in artifact.tensors:
        for suffix, part in SUFFIXES.items():
            if key.endswith("." + suffix):
                prefix = key[: -len(suffix) - 1]
                candidates.setdefault(prefix, {})[part] = key
                break
    return {
        name: parts
        for name, parts in candidates.items()
        if (
            "alpha" in parts
            or name.startswith(("adapter/", "encoder/"))
            or any("lora_" in key or "hada_" in key or "lokr_" in key for key in parts.values())
        )
        and (
            {"up", "down"} <= parts.keys()
            or {"a1", "a2", "b1", "b2"} <= parts.keys()
            or ("left" in parts and ("right" in parts or {"right_up", "right_down"} <= parts.keys()))
        )
    }


@dataclass
class Delta:
    """Own one layer's CPU factors and expose bounded rows of the effective update."""

    kind: str
    factors: dict[str, torch.Tensor]
    scale: float
    rank: int
    shape: tuple[int, int]

    def rows(self, start: int, end: int) -> torch.Tensor:
        f = self.factors
        if self.kind in {"lora", "encoder-lora"}:
            return (f["up"][start:end] @ f["down"]) * self.scale
        if self.kind == "loha":
            return (f["a1"][start:end] @ f["b1"]) * (f["a2"][start:end] @ f["b2"]) * self.scale
        right = f["right"] if "right" in f else f["right_up"] @ f["right_down"]
        indices = torch.arange(start, end)
        return (f["left"][indices // right.shape[0], :, None] * right[indices % right.shape[0], None, :]).reshape(
            end - start, -1
        ) * self.scale

    def low_rank(self, budget: int):
        """Return exact product factors when expansion fits the explicit work budget."""
        f = self.factors
        if self.kind in {"lora", "encoder-lora"}:
            return f["up"] * self.scale, f["down"]
        if self.kind == "loha" and self.rank**2 <= 1024 and sum(self.shape) * self.rank**2 <= budget:
            left = (f["a1"][:, :, None] * f["a2"][:, None, :]).reshape(self.shape[0], -1)
            right = (f["b1"][:, None, :] * f["b2"][None, :, :]).reshape(-1, self.shape[1])
            return left * self.scale, right
        return None


def load_delta(artifact: Artifact, name: str, parts: dict[str, str]) -> Delta:
    """Reject ambiguous scaling and geometry rather than report an invented update."""
    factors = {part: artifact.read(key).double() for part, key in parts.items() if part != "alpha"}
    if any(not torch.isfinite(value).all() for value in factors.values()):
        raise ValueError("nonfinite adapter factors")
    encoder = name.startswith("encoder/")
    if {"up", "down"} <= factors.keys():
        kind, rank = ("encoder-lora" if encoder else "lora"), factors["down"].shape[0]
        shape = factors["up"].shape[0], math.prod(factors["down"].shape[1:])
        factors["up"] = factors["up"].reshape(shape[0], -1)
        factors["down"] = factors["down"].reshape(rank, -1)
        if factors["up"].shape[1] != rank:
            raise ValueError("adapter up/down dimensions disagree")
    elif "a1" in factors:
        kind, rank = "loha", factors["b1"].shape[0]
        shape = factors["a1"].shape[0], factors["b1"].shape[1]
        if factors["a1"].shape != factors["a2"].shape or factors["b1"].shape != factors["b2"].shape:
            raise ValueError("LoHa factor dimensions disagree")
        if factors["a1"].shape[1] != rank:
            raise ValueError("LoHa product dimensions disagree")
    else:
        kind = "lokr-full" if "right" in factors else "lokr"
        right_shape = (
            factors["right"].shape
            if "right" in factors
            else (factors["right_up"].shape[0], factors["right_down"].shape[1])
        )
        rank = factors["right_down"].shape[0] if "right_down" in factors else 0
        if rank and factors["right_up"].shape[1] != rank:
            raise ValueError("LoKr product dimensions disagree")
        shape = factors["left"].shape[0] * right_shape[0], factors["left"].shape[1] * right_shape[1]
    if min(shape) <= 0 or (kind != "lokr-full" and rank <= 0):
        raise ValueError("empty adapter geometry")
    if encoder or kind == "lokr-full":
        scale = 1.0
    else:
        alpha = artifact.read(parts["alpha"]).item() if "alpha" in parts else artifact.metadata.get("network_alpha")
        if alpha is None:
            raise ValueError("alpha missing: effective update scale is unknown")
        scale = float(alpha) / rank
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("adapter alpha must be finite and positive")
    return Delta(kind, factors, scale, rank, shape)


def product_spectrum(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Use reduced QR to obtain exact nonzero singular values without a dense update."""
    r_left = torch.linalg.qr(left, mode="r").R
    r_right = torch.linalg.qr(right.T, mode="r").R
    return torch.linalg.svdvals(r_left @ r_right.T)


def summarize(delta: Delta, budget: int) -> dict:
    """Report exact update norm and budgeted spectrum; label all omitted computations."""
    low_rank = delta.low_rank(budget)
    values = None
    if low_rank is not None and sum(t.numel() for t in low_rank) <= budget and low_rank[0].shape[1] <= 1024:
        values = product_spectrum(*low_rank)
        norm = values.norm().item()
    elif delta.kind.startswith("lokr"):
        f = delta.factors
        right = f["right"] if "right" in f else f["right_up"] @ f["right_down"]
        norm = f["left"].norm().item() * right.norm().item() * abs(delta.scale)
        if f["left"].numel() + right.numel() <= budget and max(min(f["left"].shape), min(right.shape)) <= 1024:
            values = (torch.linalg.svdvals(f["left"])[:, None] * torch.linalg.svdvals(right)[None, :]).flatten() * abs(
                delta.scale
            )
    else:
        squared = 0.0
        rows = max(1, CHUNK // delta.shape[1])
        for start in range(0, delta.shape[0], rows):
            squared += delta.rows(start, min(start + rows, delta.shape[0])).square().sum().item()
        norm = math.sqrt(squared)
    return {
        "algorithm": delta.kind,
        "shape": list(delta.shape),
        "rank": delta.rank or None,
        "scale": delta.scale,
        "norm": norm,
        "rms": norm / math.sqrt(math.prod(delta.shape)),
        "spectrum": spectrum(values) if values is not None else {"status": "skipped_budget"},
    }


def compare_deltas(current: Delta, previous: Delta, budget: int) -> dict:
    """Compare effective matrices, invariant to invertible LoRA factor rescalings."""
    if current.shape != previous.shape:
        return {"status": "shape_mismatch"}
    a, b = current.low_rank(budget), previous.low_rank(budget)
    if (
        a is not None
        and b is not None
        and sum(t.numel() for t in (*a, *b)) <= budget
        and a[0].shape[1] + b[0].shape[1] <= 1024
    ):
        difference = product_spectrum(torch.cat((a[0], -b[0]), dim=1), torch.cat((a[1], b[1]), dim=0)).norm().item()
        norm_a, norm_b = product_spectrum(*a).norm().item(), product_spectrum(*b).norm().item()
    else:
        squared = square_a = square_b = 0.0
        rows = max(1, CHUNK // current.shape[1])
        for start in range(0, current.shape[0], rows):
            x, y = (
                current.rows(start, min(start + rows, current.shape[0])),
                previous.rows(start, min(start + rows, current.shape[0])),
            )
            squared += (x - y).square().sum().item()
            square_a += x.square().sum().item()
            square_b += y.square().sum().item()
        difference, norm_a, norm_b = math.sqrt(squared), math.sqrt(square_a), math.sqrt(square_b)
    return {
        "status": "matched",
        "delta_norm": difference,
        "reference_norm": norm_b,
        "relative_delta": difference / norm_b if norm_b else None,
        "relative_reason": None if norm_b else "zero_reference_norm",
        "cosine": max(-1.0, min(1.0, (norm_a**2 + norm_b**2 - difference**2) / (2 * norm_a * norm_b)))
        if norm_a and norm_b
        else None,
    }
