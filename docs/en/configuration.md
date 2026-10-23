# TOML and command-line configuration

[简体中文](../configuration.md) · [Documentation](README.md)

Organize configuration, override fields, and choose the right command. See [parameters](training/parameters.md) for types, defaults, and model constraints.

<details>
<summary>On this page</summary>

- [Precedence](#precedence)
- [Table responsibilities](#table-responsibilities)
- [TOML syntax](#toml-syntax)
- [CLI fields and complex values](#cli-fields-and-complex-values)
- [Command reference](#command-reference)
- [After changing settings](#after-changing-settings)

</details>

## Precedence

**Defaults → TOML → CLI overrides**. TOML is optional if the CLI supplies every required field.

```bash
uv run --no-dev unitunner train train.toml --device cuda --train.learning-rate 5e-5
uv run --no-dev unitunner train --config train.toml --device cuda --train.learning-rate 5e-5
```

Both forms select the same file. Relative paths resolve from the working directory, not the TOML directory. Use absolute asset/data paths when launching from different directories.

## Table responsibilities

| Table | Contents | Used by |
| --- | --- | --- |
| `model` | Family, variant, weights, encoding assets | Training, caching, generation, restoration |
| `model.text` | Text precision, quantization, tokenizers/processors | Encoding and text-conditioned inference |
| `dataset.defaults` | Shared dataset defaults | Training, caching |
| `dataset.groups.NAME` | Independent source, cache, geometry, sampling | Training, caching |
| `train` / `train.adapter` | Updates, optimizer, trainable scope | Training |
| `method` / `losses` | Objective, auxiliary objectives, restoration losses | Training |
| `execution` | Compute and memory policies | Training |
| `preview` / `output` | Previews, weights, resume, logging | Training |
| `sampling` / `adapter` | Sampling controls and loaded adapter | Generation |
| `runtime` | TF32 and cuDNN controls | Commands whose schemas expose this table |

Include tables needed by the task. Image generation, video generation, restoration inference, and time calibration have distinct schemas; a training file is not a universal command input.

## TOML syntax

```toml
[train]
learning_rate = 5e-5
accumulation_steps = 4
include_layers = ["layers.*"]

[dataset.defaults]
batch_size = 1
resolution = [1024, 1024]

[dataset.groups.portraits]
kind = "image"
source = "/data/portraits"
cache_directory = "/cache/portraits"
repeats = 2
```

Merge this fragment into existing tables; do not declare the same table twice. Quote strings and use `true` / `false` for booleans. TOML has no `null`: a reference default of `null` means omit the field. Empty strings and arrays may differ from omission; for example, `prompt = ""` explicitly replaces captions with empty text.

## CLI fields and complex values

Use dot-separated field paths and hyphenated field names. `--set` handles dynamic dataset groups:

```bash
uv run --no-dev unitunner validate train.toml \
  --train.learning-rate 5e-5 \
  --set 'dataset.groups.main.batch_size=1' \
  --set 'train.include_layers=["layers.*"]'
```

Quote the complete assignment when it contains spaces, arrays, JSON objects, or shell metacharacters. See `unitunner train --help` for common fields. The `--parameter-mode full` shortcut also clears adapter configuration.

## Command reference

| Command | Input and result |
| --- | --- |
| `validate` | Checks configuration, using the training schema by default; does not load models |
| `cache` | Encodes data and replaces cache directories after success |
| `train` | Updates parameters and publishes weights and resume state |
| `generate` | Image generation with an optional adapter |
| `generate-video` | Wan video or H3 image/audio-video generation |
| `restore` | One-step image restoration |
| `mid-timestep` | Estimates restoration time from a VAE and paired images |
| `weights` | Weight import/conversion; consult its `--help` for subcommands |
| `caches import` | Imports explicitly specified external MiniMaxH3 caches |

Validate the appropriate schema:

```bash
uv run --no-dev unitunner validate image-cache.toml --for cache
uv run --no-dev unitunner validate generate.toml --for generate
uv run --no-dev unitunner validate restore.toml --for restore
uv run --no-dev unitunner validate video.toml --for generate-video
uv run --no-dev unitunner validate calibration.toml --for mid-timestep
```

Successful validation establishes legal fields and combinations. Actual loading also checks asset availability, format, keys, and shapes.

## After changing settings

| Change | Action |
| --- | --- |
| Media, captions, references, VAE, text encoding, preprocessing | Rebuild affected caches |
| Save cadence, output location, preview cadence | Check output paths and preview assets |
| Model, adapter algorithm, objective, numerical policy | Start a new run, or ensure the resume state passes identity validation |
| Inference prompt, seed, or steps | Edit generation configuration; no retraining needed |

Next: [datasets](training/datasets.md) · [parameters](training/parameters.md) · [troubleshooting](troubleshooting.md).

---

[Documentation](README.md) · [Parameter reference](training/parameters.md) · [Troubleshooting](troubleshooting.md)
