"""Plotly scientific figures with bounded pages, explicit geometry and readable labels."""

import html
import re

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = ["#3B4CC0", "#B40426", "#6788EE", "#DD6B55", "#4363A8", "#A52A3A", "#586575"]
# A coolwarm ramp matching the reference: low blue, neutral ivory, high red.
ALIGNMENT_COLORS = [
    [0, "#3B4CC0"],
    [0.125, "#6788EE"],
    [0.25, "#9ABBFF"],
    [0.375, "#C9D7F0"],
    [0.5, "#F7F7F7"],
    [0.625, "#F2CBB7"],
    [0.75, "#F7A889"],
    [0.875, "#DD6B55"],
    [1, "#B40426"],
]
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "triangle-down", "x", "pentagon", "star"]
DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
BLOCK = re.compile("^(.*?(?:blocks?|layers?)[._])(\\d+)[._](.+)$")


def short(name, width=60):
    name = str(name)
    return name if len(name) <= width else name[: width // 2 - 1] + "…" + name[-width // 2 :]


def safe(text):
    return html.escape(str(text))


def natural(name):
    return [int(p) if p.isdecimal() else p for p in re.split("(\\d+)", name)]


def clean(name):
    local = name.split("/", 1)[-1]
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        local = local.removeprefix(prefix)
    return local.removesuffix(".weight")


def compact(name):
    local = clean(name)
    match = BLOCK.match(local)
    if match:
        local = f"B{int(match[2]):02d} · {match[3]}"
    for old, new in (
        ("attention", "attn"),
        ("output", "out"),
        ("projection", "proj"),
        ("lora_down", "A"),
        ("lora_up", "B"),
    ):
        local = local.replace(old, new)
    return short(local, 38)


def selected_rows(snapshot):
    return sorted(
        snapshot["effective"] or [r for r in snapshot["rows"] if r["kind"] != "factor"],
        key=lambda r: natural(r["name"]),
    )


def label_map(snapshot):
    return {r["name"]: f"L{i:03d}  {compact(r['name'])}" for i, r in enumerate(selected_rows(snapshot), 1)}


def finish(fig, title, subtitle="", *, height=1050, width=1600):
    """Use restrained publication typography; methods and coverage belong in the caption."""
    fig.update_layout(
        template="plotly_white",
        width=width,
        height=height,
        title={"text": safe(title.split(" · ")[0]), "x": 0.5, "xanchor": "center", "font": {"size": 30}},
        font={"family": "Arial, Liberation Sans, sans-serif", "size": 26, "color": "#202020"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=COLORS,
        margin={"l": 230, "r": 80, "t": 120, "b": 170},
        hovermode="closest",
        hoverlabel={"font": {"family": "Arial, sans-serif", "size": 15}},
        legend={
            "title": {"text": ""},
            "orientation": "v",
            "x": 1.025,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top",
            "font": {"size": 24},
            "tracegroupgap": 12,
            "itemsizing": "constant",
            "itemwidth": 42,
        },
        meta={"caption": title + ". " + subtitle},
        uirevision=title,
    )
    for update in (fig.update_xaxes, fig.update_yaxes):
        update(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="#333333",
            linewidth=1,
            ticks="outside",
            tickwidth=1,
            ticklen=5,
            tickfont={"size": 24},
            title_standoff=30,
            ticklabelstandoff=12,
            automargin=True,
        )
    for index, trace in enumerate(fig.data):
        if trace.type == "scatter":
            trace.line.width = 2.5
            if not trace.legendgroup:
                trace.legendgroup = f"series-{index}"
        if trace.type == "box":
            trace.line.width = 1.8
            trace.line.color = COLORS[0]
            trace.fillcolor = "rgba(255,255,255,0)"
            trace.marker.update(size=4, opacity=0.65)
        if trace.type == "table":
            trace.header.font.size = 22
            trace.cells.font.size = 20
    names = [str(t.name or "") for t in fig.data if t.type in {"scatter", "box"} and t.showlegend is not False]
    if names:
        # Reserve physical space for the legend instead of competing with axis labels.
        def text_width(name):
            text = re.sub("<[^>]*>", "", html.unescape(name))
            return 24 * sum(0.28 if c in "ilI., ·" else 0.88 if c in "MW@" else 0.58 for c in text)

        # Include the marker, plot-to-legend gap, and an outer gutter beyond the text.
        sidebar = max(260, int(np.ceil(max(text_width(n) for n in names))) + 125)
        # Center the complete axis-label/plot/legend group, not only the data region.
        fig.update_layout(width=width + sidebar - 80, margin_l=214, margin_r=sidebar + 16)
    elif fig.data and all(trace.type == "table" for trace in fig.data):
        # Tables have no axis labels or sidebar and need symmetric page gutters.
        fig.update_layout(margin_l=155, margin_r=155)
    return fig


def module(name):
    local = clean(name)
    match = BLOCK.match(local)
    return match[3] if match else local.rsplit(".", 1)[-1]


def value_text(value):
    return "—" if value is None or not np.isfinite(value) else f"{value:.4g}"


def colorbar_header(fig, colorbar, text):
    """Give the heading its own region above the highest colorbar tick."""
    colorbar.update(
        title_text="",
        tickfont_size=22,
        thickness=22,
        x=1.04,
        y=0.45,
        len=0.86,
        tickformat=".3~g",
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.04,
        y=1,
        xshift=10,
        xanchor="left",
        yanchor="bottom",
        text=text,
        showarrow=False,
        align="left",
        font_size=22,
        width=140,
    )


def allocation_pages(snapshot):
    """Rank energy shares on a common linear axis; keep concentration statistics in the caption."""
    diagnostics = snapshot.get("diagnostics", {})
    for role in diagnostics.get("roles", []):
        rows = sorted(
            [r for r in diagnostics["layers"] if r["role"] == role["role"] and r.get("energy_fraction") is not None],
            key=lambda r: r["energy_fraction"],
            reverse=True,
        )
        if not rows or not any(r["energy_fraction"] > 0 for r in rows):
            continue
        shown = rows[:20]
        fig = go.Figure(
            go.Bar(
                x=[r["energy_fraction"] * 100 for r in shown],
                y=[safe(compact(r["name"])) for r in shown],
                orientation="h",
                marker_color=COLORS[0],
                width=0.6,
                showlegend=False,
                customdata=[safe(r["name"]) for r in shown],
                hovertemplate="%{customdata}<br>Energy share: %{x:.3f}%<extra></extra>",
            )
        )
        energy_label = "Stored weight" if role["energy_kind"] == "stored_weight" else "Update"
        fig.update_xaxes(title_text=f"Fraction of total {energy_label.lower()} energy (%)", rangemode="tozero")
        fig.update_yaxes(autorange="reversed")
        # Leave an outer gutter beyond the longest layer label, including short pages.
        left_margin = max(280, max(len(compact(r["name"])) for r in shown) * 13 + 85)
        caption = f"{role['energy_kind']}; top {len(shown)}/{len(rows)} layers. 90% energy is carried by {role['layers_for_90pct_energy']} layers; effective layer count {role['effective_layer_count']:.2f}. Remaining layers retain their share in the denominator."
        result = finish(
            fig,
            f"{energy_label} energy by layer · {snapshot['label']} · {role['role']}",
            caption,
            height=max(550, len(shown) * 44 + 300),
        )
        result.update_layout(margin_l=left_margin)
        yield result


def relationship_pages(snapshot):
    """Native bubble charts expose relationships rather than three unrelated metric columns."""
    diagnostics = snapshot.get("diagnostics", {})
    for role in diagnostics.get("roles", []):
        rows = [r for r in diagnostics["layers"] if r["role"] == role["role"]]
        is_update = role["energy_kind"] != "stored_weight"
        use_rank = any(r.get("stable_rank_fraction") is not None for r in rows)
        use_base = any(r.get("relative_delta") is not None for r in rows)
        xkey = "stable_rank_fraction" if use_rank else "rms"
        ykey = "relative_delta" if use_base else "rms"
        if xkey == ykey:
            continue
        valid = [
            r
            for r in rows
            if r.get(xkey) is not None
            and r.get(ykey) is not None
            and np.isfinite(r[xkey])
            and np.isfinite(r[ykey])
            and (r[ykey] > 0)
            and (use_rank or r[xkey] > 0)
        ]
        if len(valid) < 3:
            continue
        group_counts = {g: sum(module(r["name"]) == g for r in valid) for g in {module(r["name"]) for r in valid}}
        popular = sorted(group_counts, key=lambda g: (-group_counts[g], g))[:8]
        grouped = {}
        for r in valid:
            group = module(r["name"])
            grouped.setdefault(group if group in popular else "Other modules", []).append(r)
        fig = go.Figure()
        for i, (group, items) in enumerate(grouped.items()):
            fig.add_trace(
                go.Scatter(
                    x=[r[xkey] for r in items],
                    y=[r[ykey] for r in items],
                    mode="markers",
                    name=safe(short(group, 36)),
                    marker={
                        "size": 11,
                        "opacity": 0.85,
                        "symbol": SYMBOLS[i % len(SYMBOLS)],
                        "color": COLORS[i % len(COLORS)],
                        "line": {"width": 1, "color": "white"},
                    },
                    customdata=[
                        [safe(r["name"]), r.get("energy_fraction"), r.get("peer_log_rms_score")] for r in items
                    ],
                    hovertemplate="%{customdata[0]}<br>X: %{x:.4g}<br>Y: %{y:.4g}<br>Energy share: %{customdata[1]:.2%}<br>Peer score: %{customdata[2]:.3f}<extra>%{fullData.name}</extra>",
                )
            )
        fig.update_xaxes(
            title_text="Stable rank / LoRA capacity" if use_rank else "Update RMS" if is_update else "Weight RMS",
            type="linear" if use_rank else "log",
        )
        fig.update_yaxes(
            title_text="Change norm / base norm (log scale)"
            if use_base
            else "Update RMS (log scale)"
            if is_update
            else "Stored weight RMS (log scale)",
            type="log",
        )
        yield finish(
            fig,
            f"Layer relationships · {snapshot['label']} · {role['role']}",
            f"{len(valid)}/{len(rows)} eligible layers · fixed-size points; color and marker shape denote module · legend filters modules",
        )


def distribution_pages(snapshot):
    """Box plots compare actual distributions; tiny groups retain their observed points."""
    diagnostics = {r["name"]: r for r in snapshot.get("diagnostics", {}).get("layers", [])}
    rows = [{**r, "rms": diagnostics.get(r["name"], r).get("rms")} for r in selected_rows(snapshot)]
    rows = [r for r in rows if r.get("rms") is not None and np.isfinite(r["rms"]) and r["rms"] > 0]
    update = bool(snapshot["effective"]) or any(r.get("rms_kind") == "base_difference" for r in diagnostics.values())
    grouped = {}
    for r in rows:
        grouped.setdefault(module(r["name"]), []).append(r)
    names = sorted(grouped, key=natural)
    for start in range(0, len(names), 6):
        fig = go.Figure()
        for name in names[start : start + 6]:
            entries = grouped[name]
            fig.add_trace(
                go.Box(
                    y=[r["rms"] for r in entries],
                    name=safe(short(name, 32)),
                    boxpoints="all" if len(entries) <= 100 else "outliers",
                    jitter=0.25,
                    pointpos=0,
                    quartilemethod="linear",
                    marker_color=COLORS[0],
                    customdata=[safe(r["name"]) for r in entries],
                    hovertemplate="%{customdata}<br>RMS: %{y:.5g}<extra>%{fullData.name}</extra>",
                    showlegend=False,
                )
            )
        fig.update_yaxes(
            title_text="Update RMS (log scale)" if update else "Stored weight RMS (log scale)",
            type="log",
        )
        fig.update_xaxes(title_text="Module", tickangle=-20)
        yield finish(
            fig,
            f"Module distributions · {snapshot['label']}",
            f"Modules {start + 1}-{min(start + 6, len(names))}/{len(names)} · box = quartiles, line = median · ≤100 entries: all points; larger groups: outliers · zeros excluded",
        )


def spectrum_figure(snapshot):
    rows = [r for r in selected_rows(snapshot) if r.get("spectrum", {}).get("values") and r.get("norm", 0) > 0]
    rows = sorted(rows, key=lambda r: r["norm"], reverse=True)[:6]
    if not rows:
        return None
    labels = label_map(snapshot)
    fig = go.Figure()
    cumulative = []
    normalized = []
    for i, r in enumerate(rows):
        values = np.asarray(r["spectrum"]["values"])
        c = np.cumsum(values**2) / r["norm"] ** 2 * 100
        n = values / values[0]
        cumulative.append(c.tolist())
        normalized.append(n.tolist())
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(values) + 1)),
                y=c,
                mode="lines+markers" if len(values) < 12 else "lines",
                name=safe(labels[r["name"]]),
                line={"color": COLORS[i % len(COLORS)], "dash": DASHES[i % len(DASHES)]},
                marker={"symbol": SYMBOLS[i % len(SYMBOLS)], "size": 8},
                customdata=[safe(r["name"])] * len(values),
                hovertemplate="%{customdata}<br>Direction %{x}<br>Value %{y:.5g}<extra></extra>",
            )
        )
    fig.update_xaxes(
        title_text="Singular direction index",
        dtick=1 if max(len(r["spectrum"]["values"]) for r in rows) <= 16 else None,
    )
    fig.update_yaxes(title_text="Cumulative energy (%)", range=[0, 105])
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": -0.16,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Cumulative energy",
                        "method": "update",
                        "args": [
                            {"y": cumulative},
                            {
                                "yaxis.type": "linear",
                                "yaxis.title.text": "Cumulative energy (%)",
                                "yaxis.range": [0, 105],
                                "yaxis.autorange": False,
                            },
                        ],
                    },
                    {
                        "label": "Normalized spectrum",
                        "method": "update",
                        "args": [
                            {"y": normalized},
                            {"yaxis.type": "log", "yaxis.title.text": "σᵢ / σ₁ (log)", "yaxis.autorange": True},
                        ],
                    },
                ],
            }
        ]
    )
    result = finish(
        fig,
        f"{'Effective update spectrum' if snapshot['effective'] else 'Trained weight spectrum'} · {snapshot['label']}",
        "Up to 6 largest-norm eligible layers · at most 128 directions · buttons switch energy and normalized spectrum",
        height=1100,
    )
    result.update_layout(showlegend=True, margin_b=240, legend_title={"text": "Layers", "font": {"size": 22}})
    return result


def layer_pages(snapshot):
    """Native Table replaces redundant per-layer point charts; complete numeric pages are printable."""
    rows = selected_rows(snapshot)
    labels = label_map(snapshot)
    diagnostics = {r["name"]: r for r in snapshot.get("diagnostics", {}).get("layers", [])}
    for start in range(0, len(rows), 24):
        page = rows[start : start + 24]
        values = [
            [safe(labels[r["name"]]) for r in page],
            [value_text(r.get("rms")) for r in page],
            [value_text(r.get("base", {}).get("relative_delta")) for r in page],
            [value_text(r.get("spectrum", {}).get("stable_rank")) for r in page],
            [value_text(r.get("spectrum", {}).get("energy_rank90")) for r in page],
            [value_text(diagnostics.get(r["name"], {}).get("peer_log_rms_score")) for r in page],
        ]
        fills = [
            "#FBE8E4"
            if abs(diagnostics.get(r["name"], {}).get("peer_log_rms_score") or 0) > 3.5
            else "#FAFAFA"
            if i % 2
            else "white"
            for i, r in enumerate(page)
        ]
        fig = go.Figure(
            go.Table(
                columnwidth=[340, 120, 150, 120, 130, 130],
                header={
                    "values": [
                        "<b>Parameter</b>",
                        "<b>ΔW RMS</b>" if snapshot["effective"] else "<b>Stored RMS</b>",
                        "<b>Change / base</b>",
                        "<b>Stable rank</b>",
                        "<b>90% rank</b>",
                        "<b>Peer score</b>",
                    ],
                    "fill_color": "#E8EDF9",
                    "align": "left",
                    "height": 64,
                },
                cells={
                    "values": values,
                    "fill_color": [fills] * 6,
                    "align": "left",
                    "height": 40,
                    "font": {"size": 16},
                },
            )
        )
        result = finish(
            fig,
            f"Layer reference · {snapshot['label']}",
            f"Rows {start + 1}-{start + len(page)}/{len(rows)} · — = unavailable · orange = |peer score| > 3.5 (heuristic) · full names in labels.csv",
            height=max(360, len(page) * 40 + 300),
        )
        result.update_layout(margin_b=60)
        yield result


def matrix_figure(title, values, xlabels, ylabels, *, lo, hi, unit, original=None, names=None, colorscale=None):
    details = (
        [
            [[float(v), safe(n)] for v, n in zip(vrow, nrow, strict=True)]
            for vrow, nrow in zip(original, names, strict=True)
        ]
        if names is not None
        else original
    )
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=xlabels,
            y=ylabels,
            zmin=lo,
            zmax=hi,
            colorscale=ALIGNMENT_COLORS if colorscale is None else colorscale,
            customdata=details,
            hoverongaps=False,
            colorbar={"title": {"text": unit, "side": "right"}},
            hovertemplate="%{customdata[1]}<br>Value: %{customdata[0]:.5g}<br>log₁₀: %{z:.4f}<extra></extra>"
            if names is not None
            else "Row %{y}, column %{x}<br>Value: %{customdata:.5g}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(type="category")
    result = finish(
        fig,
        title,
        f"Color = {unit} magnitude, not positive/negative direction · blank = unavailable · zeros floored only for log colors",
        height=max(850, len(ylabels) * 30 + 330),
    )
    # Balance the left axis title against the colorbar and its tick labels.
    result.update_layout(margin_l=195, margin_r=235)
    header = "log₁₀ RMS" if "RMS" in unit else "log₁₀ ratio" if unit.startswith("log") else short(unit, 18)
    colorbar_header(result, result.data[0].colorbar, safe(header))
    return result


def block_heatmaps(snapshot):
    groups = {}
    for r in selected_rows(snapshot):
        match = BLOCK.match(clean(r["name"]))
        if match:
            groups.setdefault(match[1], []).append((int(match[2]), match[3], r))
    for prefix, entries in groups.items():
        blocks = sorted({b for b, _, _ in entries})
        modules = sorted({m for _, m, _ in entries}, key=natural)
        if len(blocks) < 2 or len(modules) < 2:
            continue
        relative = any((r.get("base", {}).get("relative_delta") is not None for _, _, r in entries))
        records = {(b, m): r for b, m, r in entries}
        values = np.array(
            [
                [
                    records.get((b, m), {}).get("base", {}).get("relative_delta")
                    if relative
                    else records.get((b, m), {}).get("rms")
                    for m in modules
                ]
                for b in blocks
            ],
            dtype=float,
        )
        logs = np.log10(np.maximum(values, 1e-12))
        finite = logs[np.isfinite(logs)]
        if not len(finite):
            continue
        lo, hi = (float(finite.min()), float(finite.max()))
        if lo == hi:
            lo, hi = (lo - 0.5, hi + 0.5)
        for a in range(0, len(blocks), 24):
            for b in range(0, len(modules), 12):
                names = [
                    [records.get((block, m), {}).get("name", "Missing") for m in modules[b : b + 12]]
                    for block in blocks[a : a + 24]
                ]
                fig = matrix_figure(
                    f"Module map · {snapshot['label']} · {prefix} · rows {a + 1}-{min(a + 24, len(blocks))}/{len(blocks)}, columns {b + 1}-{min(b + 12, len(modules))}/{len(modules)}",
                    logs[a : a + 24, b : b + 12],
                    [safe(short(m, 32)) for m in modules[b : b + 12]],
                    [str(n) for n in blocks[a : a + 24]],
                    lo=lo,
                    hi=hi,
                    unit="log₁₀(change / base)" if relative else "log₁₀(RMS)",
                    original=values[a : a + 24, b : b + 12],
                    names=names,
                )
                fig.update_xaxes(title_text="Serialized module", tickangle=-20 if len(modules) > 6 else 0)
                fig.update_yaxes(title_text="Block index")
                yield fig


def subspace(item):
    pairs = item["pairs"]
    color_min = min(float(np.log10(np.maximum(p["absolute_cosine"], 0.0001)).min()) for p in pairs)
    color_min = min(-0.01, color_min)
    color_ticks = [float(i) for i in range(int(np.ceil(color_min)), 1)]
    if len(color_ticks) < 2:
        color_ticks = [color_min, 0]
    fig = make_subplots(
        rows=1,
        cols=len(pairs),
        subplot_titles=[safe(f"{p['left']} vs {p['right']} (k={p['k']})") for p in pairs],
        horizontal_spacing=0.12,
    )
    for i, p in enumerate(pairs, 1):
        values = np.asarray(p["absolute_cosine"])
        fig.add_trace(
            go.Heatmap(
                z=np.log10(np.maximum(values, 0.0001)),
                customdata=values,
                coloraxis="coloraxis",
                x=list(range(1, len(values) + 1)),
                y=list(range(1, len(values) + 1)),
                hovertemplate="Column direction %{x}<br>Row direction %{y}<br>|cosine|: %{customdata:.5f}<extra></extra>",
            ),
            row=1,
            col=i,
        )
        fig.update_xaxes(title_text=safe(p["right"] + " direction"), constrain="domain", row=1, col=i)
        fig.update_yaxes(
            title_text=safe(p["left"] + " direction"),
            autorange="reversed",
            scaleanchor="x" if i == 1 else f"x{i}",
            constrain="domain",
            row=1,
            col=i,
        )
    fig.update_layout(
        coloraxis={
            "colorscale": ALIGNMENT_COLORS,
            "cmin": color_min,
            "cmax": 0,
            "colorbar": {
                "title": {"text": "|cosine| (log scale)", "side": "right"},
                "tickvals": color_ticks,
                "ticktext": [f"{10**v:.2g}" for v in color_ticks],
                "thickness": 22,
                "x": 1.04,
            },
        }
    )
    fig.update_annotations(font_size=22)
    coverage = "; ".join(
        f"{p['left']} vs {p['right']}: k={p['k']}, overlap={p['subspace_overlap']:.4f}, captured energy={p['left_energy_fraction']:.2%}/{p['right_energy_fraction']:.2%}"
        for p in pairs
    )
    result = finish(
        fig,
        f"Subspace alignment · {compact(item['name'])}",
        f"{item['side']} singular vectors · blue→red = low→high absolute cosine on a log scale, not signed values · shared data-dependent color minimum {10**color_min:.3g}, maximum 1 · values below 10⁻⁴ are floored for color only · equal nonzero dimension per pair · compare overlap when singular values are repeated. {coverage}",
        width=650 * len(pairs) + 120 * (len(pairs) - 1) + 360,
        height=960,
    )
    result.update_layout(margin={"l": 180, "r": 180, "t": 140, "b": 170})
    colorbar_header(result, result.layout.coloraxis.colorbar, "|cosine|<br>(log scale)")
    return result


def heads_plot(items):
    """Compare update magnitude and direction with an explicitly identified reference."""
    groups = {}
    for item in items:
        if item.get("heads"):
            key = (item["snapshot"], len(item["heads"]["rms"]), item["heads"].get("comparison", ""))
            groups.setdefault(key, []).append(item)
    for (snapshot, count, comparison), entries in groups.items():
        entries = sorted(entries, key=lambda r: natural(r["name"]))
        for metric, title in (
            ("relative_change", "Head update magnitude"),
            ("cosine_with_reference", "Head direction alignment"),
        ):
            if not any(metric in r["heads"] for r in entries):
                continue
            original = np.array(
                [[np.nan if v is None else v for v in r["heads"].get(metric, [None] * count)] for r in entries]
            )
            signed = metric == "cosine_with_reference"
            if "Merged W₀+ΔW" in comparison:
                title = "cos(ΔW, W₀+ΔW) by head" if signed else "‖ΔW‖ / ‖W₀+ΔW‖ by head"
            values = original if signed else np.log10(np.maximum(original, 1e-12))
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            if signed:
                maximum = float(np.max(np.abs(finite)))
                step = 10 ** np.floor(np.log10(maximum)) if maximum else 1
                bound = min(1.0, float(np.ceil(maximum / step) * step)) if maximum else 1.0
                lo, hi = -bound, bound
            else:
                lo, hi = float(finite.min()), float(finite.max())
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            for start in range(0, len(entries), 24):
                page = entries[start : start + 24]
                for col in range(0, count, 32):
                    fig = matrix_figure(
                        f"{title} · {snapshot}",
                        values[start : start + 24, col : col + 32],
                        [str(i) for i in range(col, min(col + 32, count))],
                        [safe(compact(r["name"])) for r in page],
                        lo=lo,
                        hi=hi,
                        unit="Signed cosine" if signed else "log₁₀(‖ΔW‖ / ‖reference‖)",
                        original=original[start : start + 24, col : col + 32],
                        colorscale=ALIGNMENT_COLORS,
                    )
                    fig.update_xaxes(title_text="Head index (declared row groups)", nticks=16)
                    fig.update_yaxes(title_text="Layer / module")
                    left_margin = max(300, max(len(compact(r["name"])) for r in page) * 13 + 110)
                    fig.update_layout(
                        margin_l=left_margin,
                        margin_r=180,
                        width=max(920, left_margin + min(32, count - col) * 32 + 180),
                        height=max(600, len(page) * 36 + 310),
                    )
                    colorbar_header(fig, fig.data[0].colorbar, "Cosine" if signed else "log₁₀ ratio")
                    if signed:
                        fig.data[0].colorbar.update(tickvals=[lo, lo / 2, 0, hi / 2, hi])
                    meaning = (
                        "cos(ΔW, reference): blue = opposing, neutral = orthogonal, red = aligned; zero norms are undefined"
                        if signed
                        else "‖ΔW‖F / ‖reference‖F: blue→red = low→high magnitude; zero denominator is undefined; zeros are floored at 10⁻¹² for color only"
                    )
                    fig.update_layout(
                        meta={
                            "caption": f"{title}. {snapshot}. {comparison}. {meaning}. Rows {start + 1}-{start + len(page)}/{len(entries)}; heads {col}-{min(col + 32, count) - 1}/{count}. User-declared contiguous output-row groups; fused QKV layouts are not inferred. Color limits [{lo:.4g}, {hi:.4g}] are data-dependent and shared across pages of the same metric and comparison; signed cosine remains centered at zero."
                        }
                    )
                    yield fig


def collection_pages(snapshots):
    for start in range(0, len(snapshots), 24):
        page = snapshots[start : start + 24]
        norms = [
            sum(r.get("norm", 0) ** 2 for r in s["effective"] or s["rows"] if r.get("norm") is not None) ** 0.5
            for s in page
        ]
        fig = go.Figure(
            go.Table(
                header={
                    "values": ["Final export", "Selected tensors", "Effective updates", "Combined norm"],
                    "fill_color": "#E8EDF9",
                    "height": 64,
                },
                cells={
                    "values": [
                        [safe(s["label"]) for s in page],
                        [s.get("selected_tensors", "—") for s in page],
                        [len(s["effective"]) for s in page],
                        [value_text(v) for v in norms],
                    ],
                    "height": 40,
                    "fill_color": "#FAFAFA",
                },
            )
        )
        yield finish(
            fig,
            "Final export inventory",
            f"Files {start + 1}-{start + len(page)}/{len(snapshots)} · norms are not a cross-model quality ranking",
            height=max(650, len(page) * 40 + 320),
        )


def telemetry_pages(metrics):
    keys = sorted({k for m in metrics for k in m["values"]})
    panels = [
        ("Loss", [k for k in keys if k.startswith("loss/") and k.count("/") == 1]),
        ("Gradient norm", [k for k in keys if k.endswith("/grad_norm")]),
        ("Learning rate", [k for k in keys if "/lr/" in k]),
        ("CUDA memory (GiB)", [k for k in keys if k.startswith("memory/")]),
    ]
    for title, selected in panels:
        for start in range(0, len(selected), 8):
            fig = go.Figure()
            for key in selected[start : start + 8]:
                fig.add_trace(
                    go.Scatter(
                        x=[m["step"] for m in metrics],
                        y=[m["values"].get(key) for m in metrics],
                        name=safe(key),
                        mode="lines",
                        connectgaps=False,
                    )
                )
            fig.update_xaxes(title_text="Completed optimizer update", rangeslider={"visible": True, "thickness": 0.08})
            fig.update_yaxes(title_text=title)
            finish(
                fig,
                f"Training telemetry · {title}",
                f"Recorded observations, not inferred weight trajectories · series {start + 1}-{min(start + 8, len(selected))}/{len(selected)}",
                height=1000,
            )
            fig.update_layout(hovermode="x unified")
            yield fig


def generate(report):
    """Each analysis question gets an appropriate native Plotly chart, not a fixed dashboard."""
    snapshots = report["snapshots"]
    if len(snapshots) > 1:
        for i, fig in enumerate(collection_pages(snapshots), 1):
            yield (f"inventory-{i:03d}", fig)
    for i, snapshot in enumerate(snapshots, 1):
        for kind, factory in (
            ("allocation", allocation_pages),
            ("relationships", relationship_pages),
            ("distributions", distribution_pages),
            ("layers", layer_pages),
            ("modules", block_heatmaps),
        ):
            for page, fig in enumerate(factory(snapshot), 1):
                yield (f"{kind}-{i:03d}-{page:03d}", fig)
        spectrum = spectrum_figure(snapshot)
        if spectrum is not None:
            yield (f"spectrum-{i:03d}", spectrum)
    for i, item in enumerate(report.get("subspaces", []), 1):
        if item["pairs"]:
            yield (f"subspace-{i:03d}", subspace(item))
    for i, fig in enumerate(heads_plot(report.get("subspaces", [])), 1):
        yield (f"heads-{i:03d}", fig)
    for i, fig in enumerate(telemetry_pages(report["metrics"]), 1):
        yield (f"telemetry-{i:03d}", fig)
