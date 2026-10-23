# Final weight analysis

```bash
uv run unitunner analyze runs/my-run --base /models/base.safetensors
uv run unitunner analyze --list
```

The analyzer selects unnumbered portable exports in `weights/`, including separate encoder exports. Legacy root-level exports and explicit safetensors/index paths are supported. Numbered periodic exports and `resume/` states are excluded; missing final exports produce an error. The training writer may refresh the unnumbered latest export during training: this filename does not prove completion. Reports use recorded update metadata.

```python
from unitunner.analysis import analyze_run
report = analyze_run("runs/my-run", show="auto")
```

All charts use Plotly, without Matplotlib or Seaborn. Notebook calls display interactive Plotly figures with zoom and full-path hover details. Terminal calls print tables and display images where supported; headless terminals save results. Use `--show never` to disable display. Results go to `project/analysis/<timestamp>/`: PNGs, multipage PDF, CSV tables and Markdown, with no HTML or intermediate datasets.

Static PNG/PDF export uses Kaleido and requires Chrome/Chromium. Install with `uv run plotly_get_chrome -y`; Linux also needs the browser system libraries. Set `BROWSER_PATH` for an existing browser. Renderer failures retain numeric tables and report the missing dependency rather than switching plotting libraries.

<details>
<summary>On this page</summary>

- [Visuals and tables](#visuals-and-tables)
- [Scientific interpretation](#scientific-interpretation)
- [Matching and budgets](#matching-and-budgets)
- [Update versus merged LoRA weights](#update-versus-merged-lora-weights)

</details>

## Visuals and tables

Figures use a restrained paper style: white background, consistent serif typography, thin axes and a colorblind-friendly palette. Explanatory text and coverage belong in `figure-captions.md`, not inside plot panels.

- Energy ranking: top 20 layers, normalized by the complete eligible role population.
- Fixed-size scatter: rank usage versus relative baseline movement; fewer than three observations do not produce a relationship plot.
- Unfilled box plots with observation points: module RMS distributions, 10 modules per page.
- Spectral curves: up to six eligible layers. Interactive notebook buttons switch views; static exports omit buttons.
- Numeric reference tables: 24 rows per page, shorter canvases for small samples.
- Actual two-dimensional matrices: coolwarm for module, subspace and head maps, with explicit scales and consistent page limits.
- Separate telemetry plots: units are not mixed, notebook hover and range controls remain available.

The PDF preserves vector text, axes and curves; heatmap representation follows the renderer. A specific journal may impose additional dimensions, typography and format requirements.

Official references: [Plotly](https://plotly.com/python/), [Box](https://plotly.com/python/box-plots/), [Table](https://plotly.com/python/table/), [Heatmap](https://plotly.com/python/heatmaps/), [Hover](https://plotly.com/python/hover-text-and-formatting/).

## Scientific interpretation

LoRA uses effective `ΔW=(alpha/r)BA`, preserving factor-gauge invariance. Encoder and other adapters use their actual scaling/composition. `layers.csv` contains finite-aware FP64 moments, nonfinite counts, sampled quantiles, matched baseline changes and spectral diagnostics.

`diagnostics.csv` and `allocation.csv` include squared-norm energy fractions, layers needed for 90% energy, effective layer count `1/sum(p²)`, stable rank, energy rank, and LoRA rank-capacity fractions. Energy populations are separated by module role; dense exports use matched baseline differences when available, otherwise explicitly labeled stored-weight energy. Large matrices naturally contribute more energy; inspect RMS too.

LoRA capacity is `min(r,in,out)`, not a denominator for LoHa/LoKr. Peer scale scores use `0.67448975*(log10(RMS)-median)/MAD` within matching role, algorithm, shape and final module token; at least five positive finite observations and nonzero MAD are required. A score above 3.5 in magnitude is a heuristic observation, not a significance test or training failure.

Subspace overlap and principal angles are more reliable than individual singular-vector alignment for repeated singular values. Norms and ranks are descriptive: they cannot establish quality or overfitting. Rank reductions, target-layer changes and optimizer adjustments require validation and fixed-seed previews.

## Matching and budgets

Use `--base`, `--compare`, repeatable `--layer`, `--top-k`, `--top-layers`, and explicit `--key-map` when needed. Names and shapes must match; fuzzy mappings are not inferred. Encoders require their own matching VAE reference. Spectra and subspaces have explicit compute budgets; randomized SVD uses a fixed seed and reports captured energy. Packed quantization is storage-only unless exported to floating weights.

Final exports cannot recover parameter trajectories. Recorded loss/gradient/LR curves are separate telemetry, not inferred weight histories.

Sources: [LoRA](https://arxiv.org/abs/2106.09685), [LyCORIS](https://arxiv.org/abs/2309.14859), [Randomized matrix decompositions](https://arxiv.org/abs/0909.4061).

Legends use a reserved right sidebar rather than the axis-title area. Tick and legend text is 24 px, body text 26 px, and titles 30 px, scaled proportionally on export. Energy rankings reserve 44 px per row; distribution pages show at most six groups with uniform dark-blue outlines. Spectrum controls remain interactive in notebooks and are omitted from static PNG/PDF exports.

## Update versus merged LoRA weights

With a compatible `--base`, adapter subspace/head analysis compares the effective **ΔW** with **W₀+ΔW**. Original per-layer baseline statistics still compare with W₀. Singular-vector maps use absolute cross-basis cosine on a coolwarm log scale (data-dependent lower bound, floored at 10⁻⁴, to 1); equal nonzero k is used on both sides and energy coverage is recomputed at that k. Repeated singular values allow basis rotations, so principal angles and overlap are more reliable than individual diagonal entries.

Head maps show both `‖ΔW_h‖F / ‖(W₀+ΔW)_h‖F` (log magnitude) and signed `cos(ΔW_h, (W₀+ΔW)_h)` (blue opposing, neutral orthogonal, red aligned; symmetric data-dependent limits about zero). Undefined ratios/cosines are blank. These are weight statistics, not activation attention maps or head importance. Explicit `--heads` declares contiguous output-row groups; fused QKV, GQA and transposed projections are not inferred. Each page contains at most 24 layers and 32 heads with shared metric limits. Default selection covers at most three layers; use `--top-layers` or explicit `--layer` selectors to expand it. `heads.csv` identifies the comparison, and `principal-angles.csv` saves each principal angle. Full-weight analysis retains trained-versus-base semantics.

All figures use a common sans-serif publication style and blue/red palette. Artifact analysis is primary: effective ΔW for adapters, matched W_trained−W₀ for dense update RMS, and explicitly labeled stored weights when no base is available. Raw stored moments are retained in layers.csv; diagnostics.csv identifies its RMS source via rms_kind. Dense spectra remain explicitly labeled trained-weight spectra.

---

[Documentation](README.md) · [Parameter reference](training/parameters.md) · [Troubleshooting](troubleshooting.md)
