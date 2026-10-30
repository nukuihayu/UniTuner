"""Compute finite-aware exact moments and matched weight differences in bounded chunks."""

import math

import torch

CHUNK = 262144


def describe(tensor: torch.Tensor) -> dict:
    """Use FP64 population moments; distribution quantiles use a deterministic sample."""
    flat = tensor.detach().reshape(-1)
    count = finite = zeros = 0
    mean = m2 = squares = 0.0
    low, high = math.inf, -math.inf
    for start in range(0, flat.numel(), CHUNK):
        values = flat[start : start + CHUNK].to(torch.float64)
        count += values.numel()
        values = values[torch.isfinite(values)]
        n = values.numel()
        if not n:
            continue
        local = values.mean().item()
        variance = ((values - local) ** 2).sum().item()
        delta = local - mean
        m2 += variance + delta * delta * finite * n / (finite + n)
        mean += delta * n / (finite + n)
        finite += n
        zeros += (values == 0).sum().item()
        squares += values.square().sum().item()
        low, high = min(low, values.min().item()), max(high, values.max().item())
    sample_count = min(flat.numel(), 4096)
    indices = torch.linspace(0, max(0, flat.numel() - 1), sample_count, dtype=torch.float64).long()
    sample = flat[indices].to(torch.float64)
    sample = sample[torch.isfinite(sample)]
    quantiles = (
        torch.quantile(sample, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64)).tolist()
        if len(sample)
        else [None] * 3
    )
    nonzero_sample = sample[sample != 0].abs().log10()
    edges = torch.linspace(-12, 4, 65, dtype=torch.float64)
    histogram = torch.histogram(nonzero_sample, bins=edges).hist
    return {
        "count": count,
        "finite": finite,
        "nonfinite": count - finite,
        "zeros": zeros,
        "mean": mean if finite else None,
        "std": math.sqrt(max(0, m2 / finite)) if finite else None,
        "norm": math.sqrt(squares) if finite else None,
        "rms": math.sqrt(squares / finite) if finite else None,
        "min": low if finite else None,
        "max": high if finite else None,
        "zero_fraction": zeros / finite if finite else None,
        "sample_count": len(sample),
        "q01_sample": quantiles[0],
        "median_sample": quantiles[1],
        "q99_sample": quantiles[2],
        "log_abs_histogram": {
            "edges": edges.tolist(),
            "counts": histogram.tolist(),
            "underflow": int((nonzero_sample < -12).sum()),
            "overflow": int((nonzero_sample > 4).sum()),
        },
    }


def compare(current: torch.Tensor, reference: torch.Tensor) -> dict:
    """Compare all corresponding finite pairs; zero denominators stay undefined."""
    if current.shape != reference.shape:
        return {"status": "shape_mismatch", "reference_shape": list(reference.shape)}
    left, right = current.reshape(-1), reference.reshape(-1)
    current_sq = reference_sq = difference_sq = dot = 0.0
    finite = changed = 0
    for start in range(0, left.numel(), CHUNK):
        a, b = left[start : start + CHUNK].double(), right[start : start + CHUNK].double()
        mask = torch.isfinite(a) & torch.isfinite(b)
        a, b = a[mask], b[mask]
        finite += a.numel()
        changed += (a != b).sum().item()
        current_sq += a.square().sum().item()
        reference_sq += b.square().sum().item()
        difference_sq += (a - b).square().sum().item()
        dot += (a * b).sum().item()
    delta, base = math.sqrt(difference_sq), math.sqrt(reference_sq)
    return {
        "status": "matched" if finite == left.numel() else "nonfinite_pairs_excluded",
        "finite_pairs": finite,
        "excluded_pairs": left.numel() - finite,
        "delta_norm": delta,
        "reference_norm": base,
        "relative_delta": delta / base if base else None,
        "relative_reason": None if base else "zero_reference_norm",
        "cosine": max(-1.0, min(1.0, dot / math.sqrt(current_sq * reference_sq)))
        if current_sq and reference_sq
        else None,
        "changed_fraction": changed / finite if finite else None,
    }


def spectrum(values: torch.Tensor) -> dict:
    """Summarize the singular-value energy distribution; no universal quality score."""
    values = values.double().abs().sort(descending=True).values
    energy = values.square()
    total = energy.sum().item()
    if not total:
        return {"status": "zero", "stable_rank": 0.0, "energy_rank90": 0, "energy_effective_rank": 0.0, "values": []}
    p = energy / total
    positive = p[p > 0]
    return {
        "status": "exact",
        "stable_rank": total / energy[0].item(),
        "energy_rank90": int(torch.searchsorted(p.cumsum(0), 0.9).item()) + 1,
        "energy_effective_rank": math.exp(-(positive * positive.log()).sum().item()),
        "values": values[:128].tolist(),
        "value_count": len(values),
        "top_energy_fraction": p[0].item(),
        "displayed_energy_fraction": p[:128].sum().item(),
    }


def dense_spectrum(tensor: torch.Tensor, limit: int) -> dict:
    """Flatten convolution kernels as output by remaining axes; cap exact SVD work."""
    if tensor.ndim < 2:
        return {"status": "not_matrix"}
    matrix = tensor.reshape(tensor.shape[0], -1)
    if matrix.numel() == 0:
        return {"status": "empty"}
    if matrix.numel() > limit or min(matrix.shape) > 1024:
        return {"status": "skipped_budget", "matrix_shape": list(matrix.shape)}
    if not torch.isfinite(matrix.float()).all():
        return {"status": "nonfinite"}
    return spectrum(torch.linalg.svdvals(matrix.double()))
