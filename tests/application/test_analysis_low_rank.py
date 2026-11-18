"""Verify QR-core analysis against dense oracles without allowing dense fallbacks."""

import pytest
import torch

from unituner.analysis.adapters import Delta, compare_deltas, summarize
from unituner.analysis.artifacts import inspect_artifact
from unituner.analysis.low_rank import FactorMatrix, product_basis, product_core
from unituner.analysis.subspaces import align, declared_heads, enrich_subspaces, singular_basis
from unituner.artifacts.tensors import save_tensors


@pytest.mark.parametrize("shape,rank", [((19, 13), 4), ((5, 3), 8), ((13, 19), 4)])
@pytest.mark.parametrize("side", ["left", "right"])
def test_qr_core_rectangular_and_rank_deficient(shape, rank, side):
    generator = torch.Generator().manual_seed(91)
    left = torch.randn(shape[0], rank, dtype=torch.float64, generator=generator)
    right = torch.randn(rank, shape[1], dtype=torch.float64, generator=generator)
    left[:, -1] = left[:, 0]
    dense = left @ right
    expected = singular_basis(dense, top_k=3, side=side)
    actual = product_basis(left, right, top_k=3, side=side)
    assert actual["total_energy"] == pytest.approx(dense.square().sum().item())
    assert torch.allclose(actual["values"], expected["values"], atol=1e-12, rtol=1e-12)
    assert align(actual, expected)["subspace_overlap"] == pytest.approx(1)
    core_values, dense_values = torch.linalg.svdvals(product_core(left, right)), torch.linalg.svdvals(dense)
    assert torch.allclose(core_values, dense_values[: len(core_values)], atol=1e-12)
    assert torch.all(dense_values[len(core_values) :] < 1e-12)


def test_low_budget_norm_and_nearby_checkpoint_never_expand(monkeypatch):
    generator = torch.Generator().manual_seed(92)
    up = torch.randn(128, 4, dtype=torch.float64, generator=generator)
    down = torch.randn(4, 192, dtype=torch.float64, generator=generator)
    previous = Delta("lora", {"up": up, "down": down}, 1.5, 4, (128, 192))
    changed = up + 1e-9 * torch.randn(up.shape, dtype=up.dtype, generator=generator)
    current = Delta("lora", {"up": changed, "down": down}, 1.5, 4, (128, 192))
    oracle = 1.5 * ((changed - up) @ down)

    def forbidden(*args, **kwargs):
        pytest.fail("LoRA metrics must not materialize delta rows or compute SVD when spectrum is skipped")

    monkeypatch.setattr(Delta, "rows", forbidden)
    monkeypatch.setattr(torch.linalg, "svdvals", forbidden)
    assert summarize(previous, 1)["norm"] == pytest.approx((1.5 * (up @ down)).norm().item())
    result = compare_deltas(current, previous, 1)
    assert result["delta_norm"] == pytest.approx(oracle.norm().item(), rel=1e-6)
    assert result["cosine"] == pytest.approx(1)


def test_factored_head_and_projection_statistics_match_dense():
    generator = torch.Generator().manual_seed(93)
    a, b = [
        FactorMatrix(
            torch.randn(24, rank, dtype=torch.float64, generator=generator),
            torch.randn(rank, 32, dtype=torch.float64, generator=generator),
        )
        for rank in (3, 5)
    ]
    dense_a, dense_b = a.rows(0, 24), b.rows(0, 24)
    actual, expected = declared_heads(a, b, 4), declared_heads(dense_a, dense_b, 4)
    for key in ("relative_change", "rms", "cosine_with_reference"):
        assert actual[key] == pytest.approx(expected[key])
    for side in ("left", "right"):
        vectors = singular_basis(b, top_k=3, side=side)["vectors"]
        change = a.subtract(b)
        projection = (dense_a - dense_b) @ vectors if side == "right" else vectors.T @ (dense_a - dense_b)
        assert change.projected_energy(vectors, side) == pytest.approx(projection.square().sum().item())
    with pytest.raises(ValueError, match="Zero matrix"):
        product_basis(torch.zeros(24, 3, dtype=torch.float64), a.right)


def test_subspace_selection_budgets_factors_instead_of_dense_update(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(94)
    artifacts, snapshots = [], []
    for index, rank in enumerate((3, 4)):
        path = tmp_path / f"export{index}.safetensors"
        save_tensors(
            {
                "x.up": torch.randn(128, rank, generator=generator),
                "x.down": torch.randn(rank, 192, generator=generator),
                "x.alpha": torch.tensor(float(rank)),
            },
            path,
        )
        artifacts.append(inspect_artifact(path))
        snapshots.append(
            {
                "file": str(path),
                "label": path.name,
                "previous": artifacts[0].path.name if index else None,
                "rows": [],
                "effective": [
                    {
                        "name": "weights/x",
                        "algorithm": "lora",
                        "norm": 1.0,
                        "rms": 1.0,
                        "rank": rank,
                        "shape": [128, 192],
                    }
                ],
            }
        )

    def forbidden(*args, **kwargs):
        pytest.fail("Factored checkpoint subspaces must not expand dense delta rows")

    monkeypatch.setattr(Delta, "rows", forbidden)
    monkeypatch.setattr(FactorMatrix, "rows", forbidden)
    report = {"snapshots": snapshots, "warnings": [], "context": {}}
    enrich_subspaces(report, artifacts, matrix_budget=2500, heads=4, top_k=2)
    item = report["subspaces"][-1]
    assert item["status"] == "available", item
    assert item["pairs"][0]["right_method"] == "exact QR-core SVD"
    assert item["pairs"][0]["left_method"] == "exact QR-core SVD"
    assert 0 <= item["change_energy_in_reference_subspace"] <= 1
    assert item["heads"] is not None
    assert not report["warnings"]
