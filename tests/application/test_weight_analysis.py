"""Validate scientific diagnostics against dense oracles and real native checkpoints."""

import asyncio
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from unitunner.adapters.loha import LoHa
from unitunner.adapters.lokr import LoKr
from unitunner.adapters.lora import LoRA
from unitunner.analysis.adapters import compare_deltas, groups, load_delta, summarize
from unitunner.analysis.artifacts import discover, inspect_artifact
from unitunner.analysis.run import analyze, analyze_run
from unitunner.analysis.statistics import compare, describe
from unitunner.analysis.subspaces import align, declared_heads, singular_basis
from unitunner.artifacts.tensors import save_tensors
from unitunner.checkpoints.saving import save_checkpoint
from unitunner.checkpoints.schema import CheckpointCursor, ResumeIdentity
from unitunner.cli.main import main


def test_exact_statistics_and_undefined_denominators():
    values = torch.tensor([1.0, 2.0, 0.0, float("inf"), float("nan")])
    result = describe(values)
    assert result["mean"] == 1
    assert result["std"] == pytest.approx(np.std([1, 2, 0]))
    assert result["norm"] == pytest.approx(5**0.5)
    assert result["nonfinite"] == 2
    assert result["zero_fraction"] == pytest.approx(1 / 3)
    comparison = compare(torch.ones(3), torch.zeros(3))
    assert comparison["delta_norm"] == pytest.approx(3**0.5)
    assert comparison["relative_delta"] is None
    assert comparison["cosine"] is None
    assert describe(torch.empty(0))["rms"] is None


@pytest.mark.parametrize("algorithm", ["lora", "loha", "lokr", "lokr-full", "encoder"])
def test_effective_adapter_metrics_match_dense_oracle(tmp_path, algorithm):
    torch.manual_seed(71)
    if algorithm in ("lora", "encoder"):
        module = LoRA(6, 4, rank=2, alpha=3.0)
    elif algorithm == "loha":
        module = LoHa(6, 4, rank=2, alpha=3.0)
    else:
        module = LoKr((2, 3), (2, 2), rank=2, alpha=3.0, full_right=algorithm == "lokr-full")
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.2)
    state = {
        "layer." + key: value
        for key, value in module.state_dict().items()
        if not (algorithm == "encoder" and key == "alpha")
    }
    path = tmp_path / "adapter.safetensors"
    save_tensors(state, path, metadata={"unitunner.encoder.targets": "[]"} if algorithm == "encoder" else None)
    with inspect_artifact(path) as artifact:
        name, parts = next(iter(groups(artifact).items()))
        delta = load_delta(artifact, name, parts)
        expected = module.up.double() @ module.down.double() if algorithm == "encoder" else module.double().delta()
        assert torch.allclose(delta.rows(0, 4), expected)
        for budget in (1, 100000):
            result = summarize(delta, budget)
            assert result["norm"] == pytest.approx(expected.norm().item(), rel=1e-10)
            if result["spectrum"]["status"] == "exact":
                singular = torch.linalg.svdvals(expected)
                assert result["spectrum"]["stable_rank"] == pytest.approx(
                    (singular.square().sum() / singular[0] ** 2).item()
                )


def test_lora_effective_change_is_gauge_invariant(tmp_path):
    a, b = torch.randn(2, 7, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
    rotation = torch.tensor([[2.0, 0.5], [0.0, 0.25]], dtype=torch.float64)
    first, second = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_tensors({"x.down": a, "x.up": b, "x.alpha": torch.tensor(2.0)}, first)
    save_tensors(
        {"x.down": torch.linalg.solve(rotation, a), "x.up": b @ rotation, "x.alpha": torch.tensor(2.0)}, second
    )
    with inspect_artifact(first) as x, inspect_artifact(second) as y:
        left = load_delta(x, "weights/x", groups(x)["weights/x"])
        right = load_delta(y, "weights/x", groups(y)["weights/x"])
        assert compare_deltas(left, right, 100000)["delta_norm"] < 1e-12
        assert not torch.allclose(x.read("weights/x.up"), y.read("weights/x.up"))


def test_subspace_invariance_and_explicit_head_geometry():
    torch.manual_seed(10)
    matrix = torch.randn(8, 6, dtype=torch.float64)
    basis = singular_basis(matrix, top_k=4)
    flipped = {**basis, "vectors": -basis["vectors"]}
    result = align(basis, flipped)
    assert result["subspace_overlap"] == pytest.approx(1)
    assert np.allclose(np.diag(result["absolute_cosine"]), 1)
    rotation = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64)).Q
    rotated = {**basis, "vectors": basis["vectors"] @ rotation}
    assert align(basis, rotated)["subspace_overlap"] == pytest.approx(1)
    with pytest.raises(ValueError, match="Zero matrix"):
        singular_basis(torch.zeros(4, 4))
    assert declared_heads(matrix * 2, matrix, 2)["relative_change"] == pytest.approx([1, 1])
    with pytest.raises(ValueError, match="divide"):
        declared_heads(matrix, matrix, 3)


def test_lora_subspaces_and_heads_compare_update_with_merged_weights(tmp_path):
    base = torch.diag(torch.tensor([8.0, 4.0, 2.0, 1.0], dtype=torch.float64))
    down = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    up = torch.tensor([[1.0], [-2.0], [0.0], [0.0]], dtype=torch.float64)
    delta = up @ down
    merged = base + delta
    baseline, adapter = tmp_path / "base.safetensors", tmp_path / "lora.safetensors"
    save_tensors({"x.weight": base}, baseline)
    save_tensors({"x.down": down, "x.up": up, "x.alpha": torch.tensor(1.0)}, adapter)
    report = analyze(adapter, base=baseline, heads=2, top_k=4)
    item = next(item for item in report["subspaces"] if item["status"] == "available")
    pair = item["pairs"][0]
    assert pair["left"] == "Merged W₀+ΔW" and pair["right"] == "LoRA ΔW"
    oracle = align(singular_basis(merged, top_k=4), singular_basis(delta, top_k=4))
    assert np.allclose(pair["absolute_cosine"], oracle["absolute_cosine"])
    assert pair["left_energy_fraction"] == pytest.approx(
        torch.linalg.svdvals(merged)[0].square().item() / merged.square().sum().item()
    )
    expected = delta.reshape(2, -1).norm(dim=1) / merged.reshape(2, -1).norm(dim=1)
    assert item["heads"]["relative_change"] == pytest.approx(expected.tolist())
    assert item["heads"]["cosine_with_reference"][0] < 0
    assert item["heads"]["cosine_with_reference"][1] is None
    from unitunner.analysis.figures import heads_plot, subspace

    assert -4 <= subspace(item).layout.coloraxis.cmin < 0
    pages = list(heads_plot([item]))
    assert len(pages) == 2
    assert pages[1].data[0].zmin == -pages[1].data[0].zmax
    assert pages[1].data[0].zmax >= abs(item["heads"]["cosine_with_reference"][0])
    assert pages[1].data[0].colorbar.title.text == ""
    assert any(a.text == "Cosine" and a.y == 1 for a in pages[1].layout.annotations)


def test_dense_artifact_distributions_use_matched_training_difference(tmp_path):
    from unitunner.analysis.figures import distribution_pages

    base = torch.full((4, 4), 100.0)
    baseline, trained = tmp_path / "base.safetensors", tmp_path / "trained.safetensors"
    save_tensors({"x.weight": base}, baseline)
    save_tensors({"x.weight": base + torch.eye(4)}, trained)
    snapshot = analyze(trained, base=baseline)["snapshots"][0]
    metric = snapshot["diagnostics"]["layers"][0]
    assert metric["rms"] == pytest.approx(0.5)
    assert metric["rms_kind"] == "base_difference"
    assert snapshot["rows"][0]["rms"] > 100
    assert list(next(distribution_pages(snapshot)).data[0].y) == pytest.approx([0.5])


@pytest.fixture
def analysis_run(tmp_path):
    project = tmp_path / "run"
    project.mkdir()
    torch.manual_seed(12)
    model = nn.Linear(6, 4, bias=False)
    model.requires_grad_(False)
    adapter = nn.ModuleList([LoRA(6, 4, rank=2, alpha=2.0)])
    discriminator = nn.Linear(4, 1)
    modules = {"model": model, "adapter": adapter, "discriminator": discriminator}
    optimizer = torch.optim.AdamW([*adapter.parameters(), *discriminator.parameters()])
    identity = ResumeIdentity(config_sha256="a" * 64, dataset_manifest_sha256="b" * 64, model_artifact_sha256="c" * 64)
    for step in (1, 4):
        with torch.no_grad():
            adapter[0].up.add_(0.1)
        save_checkpoint(
            project / f"resume/model.state-{step:08d}.safetensors",
            modules,
            optimizer,
            CheckpointCursor(updates=step, microbatches=step, samples=step),
            identity,
        )
    (project / "weights").mkdir()
    save_tensors(
        {
            "layer.lora_down.weight": adapter[0].down,
            "layer.lora_up.weight": adapter[0].up,
            "layer.alpha": torch.tensor(2.0),
        },
        project / "weights/model.safetensors",
    )
    base = tmp_path / "base.safetensors"
    save_tensors({"layer.weight": model.weight}, base)
    return project, base


def test_final_exports_exclude_resume_and_periodic_weights(analysis_run):
    project, base = analysis_run
    save_tensors({"junk": torch.ones(2)}, project / "weights/model-00000004.safetensors")
    report = analyze(project, base=base, heads=2)
    assert len(report["snapshots"]) == 1
    snapshot = report["snapshots"][0]
    assert snapshot["label"] == "model.safetensors"
    assert snapshot["previous"] is None
    assert snapshot["effective"][0]["base"]["status"] == "matched"
    assert any(item["pairs"] for item in report["subspaces"])
    with pytest.raises(ValueError, match="resume"):
        discover(project / "resume")
    with pytest.raises(ValueError, match="periodic"):
        discover(project / "weights/model-00000004.safetensors")
    native = inspect_artifact(project / "resume/model.state-00000004.safetensors")
    assert native.checkpoint
    assert all(not key.startswith(("optimizer/", "rng/")) for key in native.tensors)


def test_cli_and_notebook_save_only_final_scientific_products(analysis_run, tmp_path, monkeypatch):
    project, base = analysis_run
    mapping = tmp_path / "keys.json"
    mapping.write_text(json.dumps({"weights/layer": "weights/layer.weight"}))
    output = tmp_path / "results"
    displayed = []
    monkeypatch.setattr(
        "unitunner.analysis.publication.display_image", lambda path, mode, first: displayed.append(path)
    )
    assert (
        main(
            [
                "analyze",
                str(project),
                "--base",
                str(base),
                "--key-map",
                str(mapping),
                "--heads",
                "2",
                "--output",
                str(output),
                "--show",
                "auto",
            ]
        )
        == 0
    )
    assert (output / "figures.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "layers.csv").is_file() and (output / "summary.md").is_file()
    assert any(output.glob("subspace-*.png")) and any(output.glob("heads-*.png"))
    assert displayed and all(path.exists() for path in displayed)
    assert not list(output.glob("*.html")) and not list(output.glob("*.json"))
    assert {p.suffix for p in output.iterdir()} <= {".png", ".pdf", ".csv", ".md"}
    single = tmp_path / "single.safetensors"
    save_tensors({"x.weight": torch.eye(2)}, single)
    report = analyze_run(single, output=tmp_path / "notebook", show="never")
    assert Path(report["summary_path"]).is_file()


def test_nonfinite_and_storage_only_weights_are_visible(tmp_path):
    path = tmp_path / "weights.safetensors"
    save_tensors(
        {
            "x.weight": torch.tensor([[float("nan"), 1.0]]),
            "counter": torch.tensor(2),
            "packed.weight": torch.tensor([[3]], dtype=torch.uint8),
            "packed.comfy_quant": torch.tensor([1], dtype=torch.uint8),
        },
        path,
    )
    snapshot = analyze(path)["snapshots"][0]
    assert snapshot["alerts"][0]["severity"] == "error"
    assert len(snapshot["excluded"]) == 3
    assert snapshot["rows"][0]["nonfinite"] == 1


def test_final_encoder_and_same_identity_exports_remain_independent(tmp_path):
    directory = tmp_path / "weights"
    directory.mkdir()
    metadata = {"unitunner.analysis.step": "10", "unitunner.analysis.identity": "same"}
    for name in ("a", "b"):
        save_tensors({"x.weight": torch.eye(2)}, directory / f"{name}.safetensors", metadata=metadata)
    save_tensors(
        {"x.weight": torch.eye(2)},
        directory / "a.encoder.safetensors",
        metadata={**metadata, "unitunner.encoder.targets": "[]"},
    )
    save_tensors({"x.weight": torch.eye(2)}, directory / "a-00000010.encoder.safetensors")
    report = analyze(tmp_path, base=directory / "a.safetensors")
    assert len(report["snapshots"]) == 3
    assert all(s["previous"] is None for s in report["snapshots"])
    assert any("encoder" in s["roles"] for s in report["snapshots"])
    assert len({item["snapshot"] for item in report["subspaces"]}) == 3


def test_periodic_only_directory_does_not_fall_back(tmp_path):
    (tmp_path / "weights").mkdir()
    save_tensors({"x": torch.ones(2)}, tmp_path / "weights/a-00000001.safetensors")
    with pytest.raises(ValueError, match="No final"):
        discover(tmp_path)


def test_many_layer_and_head_pages_have_bounded_rows():
    from unitunner.analysis.figures import heads_plot, layer_pages

    rows = [
        {
            "name": f"long.model.transformer.blocks.{i}.attention.output.weight",
            "rms": i + 1,
            "base": {"relative_delta": 0.01},
            "spectrum": {"stable_rank": 2},
        }
        for i in range(53)
    ]
    snapshot = {"label": "test", "effective": rows}
    pages = list(layer_pages(snapshot))
    assert len(pages) == 3
    assert [len(p.data[0].cells.values[0]) for p in pages] == [24, 24, 5]
    for page in pages:
        assert page.to_plotly_json()["data"]
    items = [
        {"snapshot": "test", "name": r["name"], "heads": {"rms": [1] * 32, "relative_change": [0.1] * 32}} for r in rows
    ]
    pages = list(heads_plot(items))
    assert len(pages) == 3
    assert len({(page.data[0].zmin, page.data[0].zmax) for page in pages}) == 1
    for page in pages:
        assert page.to_plotly_json()["data"]


def test_readable_labels_natural_order_and_real_module_geometry():
    from unitunner.analysis.figures import block_heatmaps, label_map, selected_rows

    rows = [
        {"name": f"weights/blocks.{i}.attention.{module}", "rms": 0.01 * (i + 1), "norm": 1}
        for i in (10, 2, 1)
        for module in ("output", "query")
    ]
    snapshot = {"label": "test", "effective": rows}
    labels = label_map(snapshot)
    assert "B01" in next(iter(labels.values()))
    assert "B02" in list(labels.values())[2]
    assert all(len(label) < 48 for label in labels.values())
    assert len(set(labels.values())) == len(rows)
    assert "blocks.1." in selected_rows(snapshot)[0]["name"]
    pages = list(block_heatmaps(snapshot))
    assert len(pages) == 1
    assert np.asarray(pages[0].data[0].z).shape == (3, 2)
    assert pages[0].to_plotly_json()["data"]


def test_many_final_files_are_paginated_without_series_deduplication():
    from unitunner.analysis.figures import collection_pages

    snapshots = [{"label": f"adapter-{i}", "effective": [{"norm": 1.0}], "rows": []} for i in range(49)]
    pages = list(collection_pages(snapshots))
    assert [len(fig.data[0].cells.values[0]) for fig in pages] == [24, 24, 1]
    for fig in pages:
        assert fig.to_plotly_json()["data"]


def test_energy_allocation_rank_capacity_and_peer_scores():
    from unitunner.analysis.diagnostics import enrich_diagnostics

    rows = [
        {
            "name": f"weights/blocks.{i}.out",
            "role": "weights",
            "algorithm": "lora",
            "shape": [4, 8],
            "rank": 8,
            "norm": v,
            "rms": v / 32**0.5,
            "spectrum": {"stable_rank": 2.0, "energy_rank90": 3, "energy_effective_rank": 2.5},
        }
        for i, v in enumerate([1.0, 1.1, 1.2, 1.3, 1.4, 100.0])
    ]
    report = {"snapshots": [{"effective": rows, "rows": [], "alerts": []}], "methodology": []}
    enrich_diagnostics(report)
    diagnostics = report["snapshots"][0]["diagnostics"]
    assert diagnostics["layers"][0]["lora_rank_capacity"] == 4
    assert diagnostics["layers"][0]["stable_rank_fraction"] == 0.5
    assert diagnostics["layers"][0]["rank90_fraction"] == 0.75
    assert sum(r["energy_fraction"] for r in diagnostics["layers"]) == pytest.approx(1)
    role = diagnostics["roles"][0]
    assert role["layers_for_90pct_energy"] == 1
    assert 1 < role["effective_layer_count"] < 1.01
    assert report["snapshots"][0]["alerts"][-1]["name"] == "weights/blocks.5.out"


def test_zero_energy_and_non_lora_do_not_invent_capacity():
    from unitunner.analysis.diagnostics import enrich_diagnostics

    report = {
        "snapshots": [
            {
                "effective": [
                    {
                        "name": "weights/x",
                        "role": "weights",
                        "algorithm": "loha",
                        "shape": [4, 4],
                        "rank": 2,
                        "norm": 0.0,
                        "rms": 0.0,
                        "spectrum": {"stable_rank": 0},
                    }
                ],
                "rows": [],
                "alerts": [],
            }
        ],
        "methodology": [],
    }
    enrich_diagnostics(report)
    diagnostic = report["snapshots"][0]["diagnostics"]
    assert diagnostic["layers"][0]["lora_rank_capacity"] is None
    assert diagnostic["layers"][0]["peer_log_rms_score"] is None
    assert diagnostic["roles"][0]["effective_layer_count"] is None


def test_plotly_renderer_failure_keeps_numeric_products(tmp_path, monkeypatch):
    from unitunner.analysis.run import publish

    path = tmp_path / "final.safetensors"
    save_tensors({"x.weight": torch.eye(2)}, path)
    report = analyze(path)
    monkeypatch.setattr("unitunner.analysis.publication.browser_path", lambda: "/missing/chrome")

    def unavailable(*args, **kwargs):
        raise RuntimeError("Chrome not found")

    monkeypatch.setattr("kaleido.Kaleido", unavailable)
    with pytest.raises(RuntimeError, match="BROWSER_PATH"):
        publish(report, tmp_path / "output", show="never")
    assert (tmp_path / "output/layers.csv").exists()
    assert (tmp_path / "output/summary.md").exists()
    assert not (tmp_path / "output/figures.pdf").exists()


def test_analysis_downloads_missing_chrome_to_managed_cache(monkeypatch, tmp_path):
    from unitunner.analysis.publication import browser_path

    monkeypatch.delenv("BROWSER_PATH", raising=False)
    monkeypatch.setattr("unitunner.analysis.publication.Chromium.find_browser", lambda **kwargs: None)
    downloads = []
    browser = tmp_path / "chrome"
    browser.touch()

    def download(command, **kwargs):
        downloads.append(kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout=f"{browser}\n", stderr="")

    monkeypatch.setattr("unitunner.analysis.publication.subprocess.run", download)
    assert browser_path() == str(browser)
    assert downloads == [180]


@pytest.mark.parametrize("explicit", [True, False])
def test_analysis_reuses_browser_without_download(monkeypatch, explicit):
    from unitunner.analysis.publication import browser_path

    if explicit:
        monkeypatch.setenv("BROWSER_PATH", "/custom/chrome")
    else:
        monkeypatch.delenv("BROWSER_PATH", raising=False)
    monkeypatch.setattr("unitunner.analysis.publication.Chromium.find_browser", lambda **kwargs: "/cached/chrome")

    def unexpected_download(*args, **kwargs):
        pytest.fail("An available browser must not trigger a download")

    monkeypatch.setattr("unitunner.analysis.publication.subprocess.run", unexpected_download)
    assert browser_path() == ("/custom/chrome" if explicit else "/cached/chrome")


@pytest.mark.parametrize("timeout", [False, True])
def test_analysis_download_failure_keeps_numeric_products(tmp_path, monkeypatch, timeout):
    from unitunner.analysis.publication import publish

    path = tmp_path / "final.safetensors"
    save_tensors({"x.weight": torch.eye(2)}, path)
    monkeypatch.delenv("BROWSER_PATH", raising=False)
    monkeypatch.setattr("unitunner.analysis.publication.Chromium.find_browser", lambda **kwargs: None)

    def offline(command, **kwargs):
        if timeout:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="Network unavailable")

    monkeypatch.setattr("unitunner.analysis.publication.subprocess.run", offline)
    with pytest.raises(RuntimeError, match="timed out" if timeout else "Network unavailable"):
        publish(analyze(path), tmp_path / "output", show="never")
    assert (tmp_path / "output/layers.csv").exists()
    assert (tmp_path / "output/summary.md").exists()


def test_plotly_notebook_display_is_interactive(analysis_run, tmp_path, monkeypatch):
    project, _ = analysis_run
    shown = []
    monkeypatch.setattr("unitunner.analysis.publication.notebook", lambda: True)
    monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self, **kwargs: shown.append((self, kwargs)))

    async def in_kernel():
        analyze_run(project, output=tmp_path / "notebook-interactive", show="auto")

    asyncio.run(in_kernel())
    assert shown and all(kwargs["renderer"] == "plotly_mimetype" for _, kwargs in shown)
    assert all(fig.data for fig, _ in shown)


def test_native_plotly_charts_use_meaningful_encodings(analysis_run):
    from unitunner.analysis.figures import allocation_pages, relationship_pages, spectrum_figure

    project, base = analysis_run
    report = analyze(project, base=base)
    snapshot = report["snapshots"][0]
    allocation = next(iter(allocation_pages(snapshot)))
    assert [trace.type for trace in allocation.data] == ["bar"]
    assert allocation.data[-1].x[0] == pytest.approx(100)
    assert list(relationship_pages(snapshot)) == []
    spectrum = spectrum_figure(snapshot)
    buttons = spectrum.layout.updatemenus[0].buttons
    assert len(buttons) == 2
    assert buttons[1].args[1]["yaxis.type"] == "log"
    assert spectrum.layout.template.layout.plot_bgcolor == "white"


def test_allocation_ranking_retains_full_population_denominator():
    from unitunner.analysis.diagnostics import enrich_diagnostics
    from unitunner.analysis.figures import allocation_pages

    rows = [
        {
            "name": f"weights/blocks.{i}.query",
            "role": "weights",
            "algorithm": "lora",
            "shape": [4, 4],
            "rank": 2,
            "norm": float(i + 1),
            "rms": float(i + 1) / 4,
            "spectrum": {},
        }
        for i in range(100)
    ]
    snapshot = {"label": "large", "effective": rows, "rows": [], "alerts": []}
    report = {"snapshots": [snapshot], "methodology": []}
    enrich_diagnostics(report)
    chart = next(allocation_pages(snapshot))
    assert len(chart.data[0].x) == 20
    assert sum(chart.data[0].x) < 100
    assert chart.data[0].x[0] == pytest.approx(100 * 100**2 / sum(i * i for i in range(1, 101)))
    assert "top 20/100" in chart.layout.meta["caption"]


def test_spectrum_controls_have_separate_layout_region(analysis_run):
    from unitunner.analysis.figures import spectrum_figure

    project, _ = analysis_run
    fig = spectrum_figure(analyze(project)["snapshots"][0])
    assert fig.layout.updatemenus[0].y < 0
    assert fig.layout.legend.x > 1
    assert fig.layout.legend.y == 1
    assert fig.layout.margin.r >= 240
    assert fig.layout.margin.b >= 240
