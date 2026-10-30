"""Save final scientific figures/tables and display them in notebooks or supported terminals."""

import asyncio
import base64
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import kaleido
import plotly.graph_objects as go
from choreographer.browsers.chromium import Chromium
from pypdf import PdfWriter

from unitunner.artifacts.atomic import atomic_output
from unitunner.progress import status


def notebook():
    """Detect an actual kernel, not a shell subprocess launched by a notebook cell."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and getattr(shell, "kernel", None) is not None
    except ImportError:
        return False


def display_image(path, mode, *, first):
    """Display saved final images without HTML or a mandatory interactive backend."""
    if mode == "never":
        return
    if notebook():
        from IPython.display import Image, display

        display(Image(filename=str(path)))
        return
    if not sys.stdout.isatty() and mode != "always":
        return
    if os.environ.get("TERM_PROGRAM") == "iTerm.app":
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        print(f"\033]1337;File=inline=1;width=auto:{encoded}\a")
    elif os.environ.get("KITTY_WINDOW_ID"):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        for start in range(0, len(encoded), 4096):
            more = int(start + 4096 < len(encoded))
            options = f"a=T,f=100,m={more}" if start == 0 else f"m={more}"
            print(f"\033_G{options};{encoded[start : start + 4096]}\033\\", end="", flush=True)
        print()
    elif first:
        try:
            if os.name == "nt":
                os.startfile(str(path.resolve()))
            elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
                subprocess.Popen(
                    ["xdg-open", str(path.resolve())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path.resolve())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                status("[analyze] Headless terminal: figures saved; use notebook display or open PNG/PDF locally")
        except OSError as error:
            status(f"[analyze] Could not open image viewer: {error}; saved figures remain available", level="WARN")


def number(value):
    return "—" if value is None else f"{value:.5g}" if isinstance(value, float) else str(value)


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def summary(report):
    lines = [
        "# Weight analysis",
        "",
        f"Project: `{report['project']}`",
        f"Generated: {report['created']}",
        "",
        "## Snapshot coverage",
        "",
        "| Snapshot | Update | Selected / stored tensors | NaN / Inf | Effective updates | Baseline matched / unmatched |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for snapshot in report["snapshots"]:
        bad = sum(row["nonfinite"] for row in snapshot["rows"])
        coverage = snapshot["coverage"]
        lines.append(
            f"| {escape(snapshot['label'])} | {number(snapshot['step'])} | {snapshot['selected_tensors']} / {snapshot['stored_tensors']} | {bad} | {len(snapshot['effective'])} | {coverage['base_matched']} / {coverage['base_unmatched']} |"
        )
    lines += ["", "## Update allocation and rank capacity", ""]
    for snapshot in report["snapshots"]:
        for role in snapshot.get("diagnostics", {}).get("roles", []):
            lines.append(
                f"- `{escape(snapshot['label'])}` / {role['role']}: {role['energy_kind']}; coverage {role['covered_layers']}/{role['total_layers']} layers; 90% energy in {number(role['layers_for_90pct_energy'])} layers; effective layer count {number(role['effective_layer_count'])}; top-10 energy fraction {number(role['top10_energy_fraction'])}."
            )
    lines += [
        "",
        "`diagnostics.csv` contains LoRA rank-capacity usage, relative base movement and within-peer scale scores. `labels.csv` maps readable labels to complete parameter paths. Small rank usage is a hypothesis for a rank-ablation experiment, not evidence that reducing rank preserves validation quality.",
    ]
    lines += ["", "## Findings and next checks", ""]
    alerts = [(s["label"], item) for s in report["snapshots"] for item in s["alerts"]]
    if not alerts:
        lines.append(
            "No configured numerical flags were triggered. This does not establish model quality or exclude overfitting."
        )
    for label, item in alerts:
        lines.append(f"- **{item['severity']}** · `{escape(label)}` · `{escape(item['name'])}`: {item['message']}")
    lines += [
        "",
        "Use matched validation data and fixed-seed previews to decide model improvements. Low rank usage suggests testing a smaller rank at matched validation quality, not automatically reducing capacity. A large relative change suggests checking the saved interval and denominator before reducing learning rate.",
        "",
        "## Subspace comparisons",
        "",
    ]
    for item in report.get("subspaces", []):
        lines.append(f"- `{escape(item['snapshot'])}` / `{escape(item['name'])}` ({item['side']} vectors):")
        for pair in item["pairs"]:
            lines.append(
                f"  {pair['left']} vs {pair['right']}: k={pair['k']}, overlap={pair['subspace_overlap']:.6f}; captured energy {pair['left_energy_fraction']:.2%} / {pair['right_energy_fraction']:.2%}. {pair['left_method']} / {pair['right_method']}."
            )
        if item.get("reason"):
            lines.append("  Unavailable: " + item["reason"])
        if item.get("comparison_note"):
            lines.append("  " + item["comparison_note"])
    lines += ["", "## Missing data and limits", ""]
    if not report["baseline"]:
        lines.append(
            "- No baseline supplied: absolute change from initialization is unknown. Final exports alone cannot recover parameter trajectories."
        )
    if not report["metrics"]:
        lines.append(
            "- No recorded scalar training metrics found; loss/gradient trajectories cannot be reconstructed from weights."
        )
    for snapshot in report["snapshots"]:
        omitted = snapshot["omitted_frozen"]
        skipped = sum(
            row.get("spectrum", {}).get("status") == "skipped_budget"
            for row in snapshot["rows"] + snapshot["effective"]
        )
        lines.append(
            f"- {escape(snapshot['label'])}: {len(snapshot['excluded'])} storage-only/nonfloat tensors; {omitted} frozen tensors omitted from source; {skipped} spectra skipped by budget."
        )
    lines.extend("- " + warning for warning in report["warnings"])
    lines += ["", "## Methods", ""] + ["- " + item for item in report["methodology"]]
    lines += ["", "Options: `" + str(report["options"]) + "`", "", "## Sources", ""]
    lines += [f"- [{item['title']}]({item['url']})" for item in report["references"]]
    return "\n".join(lines) + "\n"


def write_csv(path, columns, rows):
    """Export final tables; guard user-derived cells against spreadsheet formula execution."""
    with atomic_output(path) as temporary, temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@")) else value
                    for key, value in row.items()
                }
            )


def terminal_table(report):
    """Show concise actual results in ordinary terminals without requiring image support."""
    print("\nWEIGHT ANALYSIS  ·  saved artifacts, CPU / FP64")
    print(f"{'Snapshot':42} {'Step':>9} {'Tensors':>9} {'NaN/Inf':>10} {'Base match':>12}")
    for snapshot in report["snapshots"]:
        bad = sum(row["nonfinite"] for row in snapshot["rows"])
        match = snapshot["coverage"]
        print(
            f"{snapshot['label'][:42]:42} {number(snapshot['step']):>9} {len(snapshot['rows']):>9} {bad:>10} {match['base_matched']:>5}/{match['base_matched'] + match['base_unmatched']:<6}"
        )
    errors = sum(item["severity"] == "error" for s in report["snapshots"] for item in s["alerts"])
    print(f"Numerical errors: {errors}. See summary.md for observations, coverage and next checks.\n")


def browser_path():
    """Reuse an installed browser or install Kaleido's cached Chrome once."""
    explicit = os.environ.get("BROWSER_PATH")
    if explicit:
        return explicit
    installed = Chromium.find_browser(skip_local=False)
    if installed:
        return installed
    status("[analyze] Chrome not found; downloading Chrome to Kaleido's user cache (first export only)")
    # Isolate the downloader so a stalled network can be stopped without changing
    # process-wide socket defaults or leaving a live download thread behind.
    result = subprocess.run(
        [sys.executable, "-c", "import kaleido; print(kaleido.get_chrome_sync())"],
        capture_output=True,
        text=True,
        timeout=180,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode:
        raise RuntimeError(f"Chrome download failed: {result.stderr.strip()[-2000:]}")
    path = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if not path or not Path(path).is_file():
        raise RuntimeError("Chrome downloader did not return an installed browser")
    return path


async def export_figures(report, directory, browser, *, interactive):
    """Own one browser and propagate startup/render errors through the awaiting caller."""
    from unitunner.analysis.figures import generate

    captions = ["# Figure captions", ""]
    displayed = []
    pdf = PdfWriter()
    try:
        async with kaleido.Kaleido(n=1, path=browser) as renderer:
            for name, fig in generate(report):
                name = re.sub(r"[^\w.-]", "_", name)
                export = go.Figure(fig)
                if export.layout.updatemenus:
                    export.layout.updatemenus = ()
                    export.update_layout(margin_b=170)
                captions.extend([f"## {name}", "", (fig.layout.meta or {}).get("caption", name), ""])
                png = await renderer.calc_fig(export, opts={"format": "png", "scale": 1.5})
                page = await renderer.calc_fig(export, opts={"format": "pdf"})
                path = directory / f"{name}.png"
                with atomic_output(path) as image:
                    image.write_bytes(png)
                pdf.append(BytesIO(page))
                status(f"[analyze] Figure saved: {path}")
                displayed.append((path, fig if interactive else None))
        with atomic_output(directory / "figure-captions.md") as temporary:
            temporary.write_text("\n".join(captions), encoding="utf-8")
        with atomic_output(directory / "figures.pdf") as temporary:
            with temporary.open("wb") as stream:
                pdf.write(stream)
    finally:
        pdf.close()
    return displayed


def publish(report: dict, directory: Path, *, show="auto") -> Path:
    """Save only final deliverables: PNG, a multipage PDF, scientific CSV tables and Markdown."""
    from unitunner.analysis.figures import label_map

    if show not in {"auto", "always", "never"}:
        raise ValueError("show must be auto, always or never")
    directory.mkdir(parents=True, exist_ok=True)
    terminal_table(report)
    labels = [
        {"snapshot": snapshot["label"], "label": label, "parameter_path": name}
        for snapshot in report["snapshots"]
        for name, label in label_map(snapshot).items()
    ]
    write_csv(directory / "labels.csv", ["snapshot", "label", "parameter_path"], labels)
    diagnostics = [
        {"snapshot": s["label"], **r} for s in report["snapshots"] for r in s.get("diagnostics", {}).get("layers", [])
    ]
    if diagnostics:
        columns = list(dict.fromkeys(k for r in diagnostics for k in r))
        write_csv(directory / "diagnostics.csv", columns, diagnostics)
    allocations = [
        {"snapshot": s["label"], **{k: v for k, v in r.items() if k != "cumulative_energy"}}
        for s in report["snapshots"]
        for r in s.get("diagnostics", {}).get("roles", [])
    ]
    if allocations:
        write_csv(directory / "allocation.csv", list(allocations[0]), allocations)
    rows, spectra = [], []
    for snapshot in report["snapshots"]:
        for row in snapshot["rows"] + snapshot["effective"]:
            rows.append(
                {
                    "snapshot": snapshot["label"],
                    "step": snapshot["step"],
                    "name": row["name"],
                    "kind": row.get("kind", "effective_update"),
                    "shape": str(row.get("shape")),
                    "dtype": row.get("dtype"),
                    "count": row.get("count"),
                    "finite": row.get("finite"),
                    "norm": row.get("norm"),
                    "rms": row.get("rms"),
                    "mean": row.get("mean"),
                    "std": row.get("std"),
                    "min": row.get("min"),
                    "max": row.get("max"),
                    "nonfinite": row.get("nonfinite"),
                    "zero_fraction": row.get("zero_fraction"),
                    "sample_count": row.get("sample_count"),
                    "q01_sample": row.get("q01_sample"),
                    "median_sample": row.get("median_sample"),
                    "q99_sample": row.get("q99_sample"),
                    "previous_status": row.get("previous", {}).get("status"),
                    "previous_delta_norm": row.get("previous", {}).get("delta_norm"),
                    "previous_relative_delta": row.get("previous", {}).get("relative_delta"),
                    "previous_cosine": row.get("previous", {}).get("cosine"),
                    "base_status": row.get("base", {}).get("status"),
                    "base_relative_delta": row.get("base", {}).get("relative_delta"),
                    "spectrum_status": row.get("spectrum", {}).get("status"),
                    "stable_rank": row.get("spectrum", {}).get("stable_rank"),
                    "energy_rank90": row.get("spectrum", {}).get("energy_rank90"),
                    "energy_effective_rank": row.get("spectrum", {}).get("energy_effective_rank"),
                    "unavailable_reason": row.get("reason"),
                }
            )
            for index, value in enumerate(row.get("spectrum", {}).get("values", []), 1):
                spectra.append(
                    {"snapshot": snapshot["label"], "name": row["name"], "index": index, "singular_value": value}
                )
    write_csv(directory / "layers.csv", list(rows[0]) if rows else ["snapshot", "name"], rows)
    write_csv(directory / "spectra.csv", ["snapshot", "name", "index", "singular_value"], spectra)
    pairs = []
    for item in report.get("subspaces", []):
        for pair in item["pairs"]:
            pairs.append(
                {
                    "snapshot": item["snapshot"],
                    "name": item["name"],
                    "side": item["side"],
                    "change_energy_in_reference_subspace": item.get("change_energy_in_reference_subspace"),
                    "left": pair["left"],
                    "right": pair["right"],
                    "k": pair["k"],
                    "overlap": pair["subspace_overlap"],
                    "max_principal_angle_deg": max(pair["principal_angles_degrees"]),
                    "mean_principal_angle_deg": sum(pair["principal_angles_degrees"]) / pair["k"],
                    "left_energy_fraction": pair["left_energy_fraction"],
                    "right_energy_fraction": pair["right_energy_fraction"],
                    "left_method": pair["left_method"],
                    "right_method": pair["right_method"],
                }
            )
    write_csv(directory / "subspaces.csv", list(pairs[0]) if pairs else ["snapshot", "name", "overlap"], pairs)
    angles = [
        {
            "snapshot": item["snapshot"],
            "name": item["name"],
            "left": pair["left"],
            "right": pair["right"],
            "direction": index,
            "principal_angle_degrees": angle,
        }
        for item in report.get("subspaces", [])
        for pair in item["pairs"]
        for index, angle in enumerate(pair["principal_angles_degrees"], 1)
    ]
    if angles:
        write_csv(directory / "principal-angles.csv", list(angles[0]), angles)
    head_rows = [
        {
            "snapshot": item["snapshot"],
            "name": item["name"],
            "declared_head": index,
            "delta_rms": rms,
            "relative_change": relative,
            "cosine_with_reference": item["heads"].get("cosine_with_reference", [None] * len(item["heads"]["rms"]))[
                index
            ],
            "comparison": item["heads"].get("comparison", ""),
        }
        for item in report.get("subspaces", [])
        if item.get("heads")
        for index, (rms, relative) in enumerate(
            zip(item["heads"]["rms"], item["heads"]["relative_change"], strict=True)
        )
    ]
    if head_rows:
        write_csv(directory / "heads.csv", list(head_rows[0]), head_rows)
    if report["metrics"]:
        metrics = [
            {"step": item["step"], "metric": key, "value": value}
            for item in report["metrics"]
            for key, value in item["values"].items()
        ]
        write_csv(directory / "training-metrics.csv", ["step", "metric", "value"], metrics)
    with atomic_output(directory / "summary.md") as temporary:
        temporary.write_text(summary(report), encoding="utf-8")
    interactive = show != "never" and notebook()
    try:
        browser = browser_path()

        def render():
            return asyncio.run(export_figures(report, directory, browser, interactive=interactive))

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            figures = render()
        else:
            # Notebook kernels own an event loop; keep ours isolated and joined.
            with ThreadPoolExecutor(max_workers=1) as executor:
                figures = executor.submit(render).result()
    except Exception as error:
        raise RuntimeError(
            "Plotly static export failed. Missing Chrome is installed automatically; "
            'if downloading is unavailable, install it with `python -c "import kaleido; kaleido.get_chrome_sync()"` '
            "or set BROWSER_PATH to an installed Chrome/Chromium. Linux may also need Chrome system libraries. "
            f"Numeric tables and summary.md are preserved in {directory}. Renderer error: {error}"
        ) from error
    for index, (path, fig) in enumerate(figures):
        if fig is not None:
            fig.show(renderer="plotly_mimetype")
        else:
            display_image(path, show, first=index == 0)
    return directory / "summary.md"
