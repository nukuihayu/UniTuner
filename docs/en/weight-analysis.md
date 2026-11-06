# Analyze trained weights

[简体中文](../weight-analysis.md) · [Documentation](README.md)

Inspect where adapter updates concentrate, check nonfinite values and compare trained weights with the base. Use fixed-prompt previews or a validation set to judge output quality.

## Generate your first report

```bash
uv run unituner analyze --list
uv run unituner analyze runs/my-run --base /models/base.safetensors
```

Replace `runs/my-run` with the training output and use the original training base for `--base`. You can also select a weight file directly:

```bash
uv run unituner analyze runs/my-run/weights/model.safetensors --base /models/base.safetensors
```

Without `--base`, weight statistics remain available but base-relative comparisons are limited. Encoder adapters need the matching VAE base.

The default selects unnumbered exports in `weights/` and analyzes encoder adapters separately. Numbered periodic exports and resume states are not accepted as analysis inputs. An unnumbered file can be the most recent export from an ongoing run; check the report's recorded update count.

## View and save reports

Results default to `analysis/<timestamp>/` inside the training directory. Choose another location with:

```bash
uv run unituner analyze runs/my-run --base /models/base.safetensors \
  --output reports/my-run --show never
```

| File | Use |
| --- | --- |
| `summary.md` | Analysis summary |
| `figures.pdf` and PNG charts | View, share or retain comparisons |
| `layers.csv` | Per-layer weight statistics |
| `diagnostics.csv`, `allocation.csv` | Update magnitudes, nonfinite values and distribution |
| `labels.csv` | Short chart labels mapped to full layer names |
| `heads.csv`, `subspaces.csv` | Results when those analyses are enabled |

`--show never` disables automatic display while still saving charts and tables. Avoid overwriting a weight file while it is being analyzed.

## Read the charts

| Chart | Look for |
| --- | --- |
| Layer energy ranking | Layers carrying most of the update |
| Update magnitude distribution | Unusually large, small or persistently zero updates among similar modules |
| Spectrum and effective rank | Adapter capacity use; useful for planning a lower-rank comparison |
| Base-relative movement | Update size relative to the original weights |
| Training log curves | Loss, learning rate, gradients and memory over time |

LoRA reports analyze effective weight updates. Full-parameter reports use trained-minus-base differences when a matching base is supplied; otherwise read values labeled as stored weights accordingly.

Investigate NaN/Inf first. Large updates, low effective rank or concentrated updates alone do not establish failure or overfitting. Compare generated results too.

## Select layers and comparison scope

```bash
uv run unituner analyze runs/my-run --base /models/base.safetensors \
  --layer '*attn*' --top-k 64 --top-layers 6
```

| Option | Use |
| --- | --- |
| `--layer` | Filter layer names; repeat for multiple selectors |
| `--top-layers` | Change the number of layers analyzed in detail; default is up to 3 |
| `--top-k` | Control the number of directions in subspace comparison |
| `--compare` | Add another complete weight file |
| `--key-map` | Supply an explicit layer-name mapping |
| `--heads` | Inspect head statistics when output-row groups are known |

Head plots describe weight size and direction, not image-generation attention maps. Check grouping before analyzing fused or special head layouts. Supply floating base weights when packed quantization prevents a full logical comparison.

## Use in a notebook

```python
from unituner.analysis import analyze_run

report = analyze_run("runs/my-run", show="auto")
```

Direct Python-cell calls show interactive charts with zoom and layer-name hover. When launching through `!uv run`, open the saved report files.

## If chart export fails

PNG/PDF export requires Chrome or Chromium. The first run can download a browser automatically. Offline, set `BROWSER_PATH` to an existing browser. You can also install one in advance:

```bash
uv run plotly_get_chrome -y
```

Linux also needs the browser's system libraries. Check the error log if export fails; any generated CSV tables and `summary.md` remain usable.

---

[Documentation](README.md) · [Inference](inference.md) · [Troubleshooting](troubleshooting.md)
