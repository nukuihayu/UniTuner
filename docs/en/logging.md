# Logs and training monitoring

[简体中文](../logging.md) · [Trainer](training/README.md)

<details>
<summary>On this page</summary>

- [What appears in the terminal](#what-appears-in-the-terminal)
- [Jupyter / Notebook](#jupyter--notebook)
- [Full metrics in JSONL](#full-metrics-in-jsonl)
- [TensorBoard](#tensorboard)
- [Diagnose failures](#diagnose-failures)

</details>

## What appears in the terminal

Interactive training updates one progress display with update/total, loss, iteration time, learning rate, optional discriminator loss and GPU memory. It does not print INFO at every update. Save/preview transitions and errors appear immediately.

Background or redirected runs disable dynamic bars and emit summaries every 60 seconds, always reporting the first and final updates. This is the default; normally omit log_every_seconds. Override only when needed:

```toml
[output]
log_every_seconds = 120.0
```

Noninteractive JSON stdout follows the summary cadence; consume JSONL or TensorBoard for every-update data.

## Jupyter / Notebook

Direct calls in Python cells use a widget bar when `ipywidgets` is available, falling back to text otherwise. `%%bash` and `!python` launch separate processes that cannot directly create kernel widgets. An inherited `JPY_PARENT_PID` enables a text bar instead of disabling progress.

If your notebook environment does not pass that variable, add this before the training command in your existing `%%bash` cell:

```bash
export UNITUNNER_PROGRESS=always
```

`UNITUNNER_PROGRESS` defaults to `auto`; `always` forces a bar, while `never` disables it for redirected logs or JSON stdout consumers. Bars use stderr. When enabled, they replace duplicate training INFO/JSON summaries; JSONL and TensorBoard still retain every update. The notebook frontend controls text carriage-return rendering. Subprocess text bars do not require `ipywidgets`.

## Full metrics in JSONL

Training automatically writes logs/<run-name>.jsonl beneath the output directory. Each update retains component metrics regardless of console refresh cadence. Entries also include timestamps, level, stage/context, source location and full exception chains.

Omitting output.directory creates a name/timestamp run; an explicit output directory uses a fixed run name. Fresh runs replace their own log, while --resume appends. The CLI prints the selected paths at startup.

## TensorBoard

```toml
[output]
tensorboard = "runs/tensorboard"
```

Only the event root is required. Run names follow the training output identity. Resume continues from its restored update.

```bash
tensorboard --logdir runs/tensorboard --host 0.0.0.0 --port 6006
```

| Metric | Read it as |
| --- | --- |
| loss/main or loss/generator | Sample-weighted loss for a completed effective update |
| loss/generator/lrr, l1, dv3d, fdl, gan, histogram | Restoration components on their own scales |
| loss/discriminator | Separate discriminator objective |
| train/*/grad_norm, lr/* | Gradient norm and parameter-group learning rates |
| train/update_samples, train/update_microbatches | Actual work in the accumulated update |
| performance/* | Update time and sample throughput |
| memory/* | CUDA allocation, reservation and peak allocation |

## Diagnose failures

```bash
uv run --no-dev unitunner train train.toml --device cuda --log-level DEBUG
```

Console levels are DEBUG/INFO/WARNING/ERROR. JSONL retains DEBUG details independently. Errors identify configuration fields or the active device, source file, update and phase when available. Start with the first exception and the announced JSONL path.

Schema errors occur before weights load. Check table paths, spelling and types; do not quote numbers/booleans. A cache identity error calls for rebuilding with the intended assets/data. An OOM calls for smaller batches/dimensions, recomputation or a suitable offload policy; inspect peak memory rather than only post-update allocation.

---

[Documentation](README.md) · [Parameter reference](training/parameters.md) · [Troubleshooting](troubleshooting.md)
