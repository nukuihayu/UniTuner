# Images to LoRA

[简体中文](../getting-started.md) · [Documentation](README.md)

This tutorial trains Z-Image Base through data preparation, configuration, caching, previews and inference. For another model, copy its [example](examples/README.md) and replace the model, VAE, text encoder and variant together.

[Environment details](installation.md) · [TOML and CLI](configuration.md) · [Troubleshooting](troubleshooting.md)

<details>
<summary>On this page</summary>

- [1. Install](#1-install)
- [2. Fill model paths](#2-fill-model-paths)
- [3. Prepare images and captions](#3-prepare-images-and-captions)
- [4. Set duration, capacity and precision](#4-set-duration-capacity-and-precision)
- [5. Configure previews and saving](#5-configure-previews-and-saving)
- [6. Validate, cache and train](#6-validate-cache-and-train)
- [7. Inspect outputs and resume](#7-inspect-outputs-and-resume)
- [8. Generate with your LoRA](#8-generate-with-your-lora)

</details>

## 1. Install

Run in an environment with a compatible CUDA driver:

```bash
git clone https://github.com/nukuihayu/UniTuner.git
cd UniTuner
uv python install 3.13
uv sync --locked --no-dev --extra media --extra text
uv run --no-dev unituner --help
```

`media` installs image/video/audio I/O dependencies; `text` installs text encoding and tokenizer dependencies. Run subsequent commands from the repository root.

If you need the Tsinghua PyPI mirror:

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --locked --no-dev --extra media --extra text
```

This is a Python package index, not a model or Python-interpreter download mirror. After installation, `uv run --no-sync --no-dev` starts commands without another dependency synchronization.

## 2. Fill model paths

Prepare a denoiser, its matching VAE and text encoder weights, then copy the example:

```bash
cp docs/examples/tutorial-lora.toml train.toml
```

Edit the existing model table:

```toml
[model]
family = "z-image"
variant = "base"
weights = "/models/z-image/diffusion_models/z_image_bf16.safetensors"
vae = "/models/z-image/vae/ae.safetensors"
text_encoder = "/models/z-image/text_encoders/qwen_3_4b.safetensors"
```

Supported Diffusers/ComfyUI weights for the family can be used directly. No trainer-specific conversion is required. Match the variant to the checkpoint; Turbo weights require turbo and matching preview sampling settings.

Tokenizer files are separate from encoder weights. The family default source provides tokenizer assets. For local tokenizer files add:

```toml
[model.text.tokenizer]
source = "/models/z-image/tokenizer"
offline = true
```

Point directly at the tokenizer files. If using a repository root with assets in a subdirectory, add `subfolder = "tokenizer"`. Downloading tokenizer assets does not download text encoder weights.

## 3. Prepare images and captions

```text
data/garden/
├── 001.jpg
├── 001.txt
├── 002.png
└── 002.txt
```

Images live directly in the source directory. Same-stem UTF-8 txt files describe each image, for example `a garden path, stone steps, flowers, gardenstyle`. Missing sidecars produce empty captions.

Edit the data table:

```toml
[dataset.groups.main]
kind = "image"
source = "data/garden"
cache_directory = "cache/garden"
resolution = [1024, 1024]
batch_size = 1
repeats = 1
enable_bucket = true
```

Resolution is [width, height]. With buckets enabled, it defines target area while actual crop dimensions follow aspect ratio. Each group needs its own cache directory.

For a shared caption set `prompt = "a garden in gardenstyle"` in this table. It overrides existing sidecars too; `prompt = ""` explicitly selects empty captions. Remove prompt to use per-image txt files. See [datasets](training/datasets.md) for multiple groups and references.

## 4. Set duration, capacity and precision

Merge these values into the existing tables:

```toml
[train]
parameter_mode = "adapter"
learning_rate = 1e-4
max_updates = 100
accumulation_steps = 1
seed = 42

[train.adapter]
algorithm = "lora"
rank = 32
alpha = 32.0

[execution]
precision = "bf16"
recompute = "block"
```

This is a CUDA BF16 starting recipe. Use fp32 for CPU checks. auto follows the checkpoint dtype rather than selecting a device-dependent fallback. If memory permits, compare stable throughput with recompute=none.

| Goal | Field | How to decide |
| --- | --- | --- |
| Train longer | max_updates | Use fixed-prompt previews to detect overfitting; start with a short run |
| Increase adapter capacity | rank, alpha | Usually start with alpha=rank and compare after changing both |
| Reduce forward memory | batch_size | Reduce to 1 before increasing accumulation |
| Increase effective batch | accumulation_steps | Approximately batch_size × accumulation_steps for one group; each update takes longer |
| Reduce update magnitude | learning_rate | Compare details, color and generalization, not total loss alone |

## 5. Configure previews and saving

```toml
[output]
directory = "runs/garden"
name = "garden"
save_every_updates = 50
tensorboard = "runs/tensorboard"

[preview]
every_updates = 50
at_start = true

[[preview.samples]]
prompt = "a garden path in gardenstyle, soft morning light"
width = 1024
height = 1024
steps = 20
guidance = 4.0
seed = 42
```

The steps/guidance values belong to this Base preview recipe; do not copy them unchanged to Turbo. Previews need a VAE and text encoder. Keep prompts and seeds fixed when comparing training versions.

Omitting output.directory creates a directory with a name and timestamp. An explicit path is reused: a fresh run clears its recognized same-name training artifacts, while resume preserves them.

Defaults retain one resume snapshot and all inference weight versions. No extra keys are needed. Use max_resume_checkpoints for more resume history, or max_weight_checkpoints to cap weight sets; 0 means unlimited for either.

## 6. Validate, cache and train

```bash
uv run --no-dev unituner validate train.toml
uv run --no-dev unituner cache train.toml --device cuda
uv run --no-dev unituner train train.toml --device cuda
```

validate checks fields and combinations without loading weights. cache loads the VAE/text encoder and writes latent/text features. Rebuild after editing images, captions, encoders or preprocessing.

Each cache invocation re-encodes all inputs and replaces the previous cache after success; failures preserve it. Do not train against a cache while rebuilding it.

Z-Image can batch VAE and text encoding with `uv run --no-dev unituner cache train.toml --device cuda --cache.batch-size 16`, or `[cache] batch_size = 16` in TOML. This does not change the training batch size. The default is 1; reduce it if device memory is insufficient. Other families currently require 1.

Terminals refresh progress and metrics in place. Background/redirected runs summarize every 60 seconds, including the first/final updates. Usually omit log_every_seconds. JSONL and enabled TensorBoard retain every component loss, gradient norm, learning rate, timing and memory metric.

## 7. Inspect outputs and resume

After the 100-update example:

```text
runs/garden/
├── weights/
│   ├── garden-00000050.safetensors
│   ├── garden-00000100.safetensors
│   └── garden.safetensors
├── resume/
│   └── garden.state-00000100.safetensors
├── previews/
└── logs/garden.jsonl
```

garden.safetensors tracks every export. Numbered files retain training versions. Only resume/ state files provide exact continuation; ordinary LoRA files have no optimizer or RNG state.

```bash
uv run --no-dev unituner train train.toml --device cuda --resume runs/garden/resume/garden.state-00000100.safetensors
tensorboard --logdir runs/tensorboard --host 0.0.0.0 --port 6006
```

Resume continues toward the configured total max_updates; it does not add that many new updates. Resuming an already completed final step performs no new updates. Models, data and numerical settings must pass resume identity checks.

## 8. Generate with your LoRA

```bash
cp docs/examples/tutorial-generate.toml generate.toml
uv run --no-dev unituner generate generate.toml --device cuda
```

Fill the same base/VAE/text paths. Set adapter.weights to runs/garden/weights/garden.safetensors and match algorithm/rank/alpha to training. Edit sampling prompts, dimensions and seed. Check the output path to avoid overwriting input images.

Next: [parameter reference](training/parameters.md), [memory and performance](training/execution.md), [restoration](restoration-tutorial.md).

---

[Documentation](README.md) · [Parameter reference](training/parameters.md) · [Troubleshooting](troubleshooting.md)
