<p align="center">
  <img src="docs/assets/unituner-logo.png" alt="UniTuner" width="720">
</p>
<p align="center"><strong>Native training for image, video, and audio-video models</strong></p>
<p align="center">English · <a href="README.zh-CN.md">简体中文</a></p>
<p align="center">
  <a href="docs/en/getting-started.md">Quick start</a> ·
  <a href="docs/en/README.md">Documentation</a> ·
  <a href="docs/en/training/models.md">Models</a> ·
  <a href="docs/en/training/parameters.md">Parameters</a> ·
  <a href="docs/en/examples/README.md">Examples</a>
</p>

---

UniTuner connects data encoding, fine-tuning, previews, and weight publication in one configurable workflow. Train **LoRA, LoHa, LoKr, or full model parameters** for subjects, styles, and conditional tasks. Use paired images or synthetic degradation to train a one-step image restoration model.

Keep complete recipes in TOML and override individual settings from the CLI. Model assets, datasets, objectives, and execution policies are configured separately, while each family retains its native conditioning and sampling semantics.

**Explore:** [About](#about-the-project) · [Models](#supported-models) · [Workflow](#from-data-to-a-model) · [Memory](#memory-precision-and-execution) · [Quick start](#quick-start)

## About the project

UniTuner is for fine-tuning with local model assets, covering the path from preparing a dataset to publishing weights for inference. Train subject or style adapters, learn relationships between reference and target images for editing, prepare temporal conditions for video and audio-video models, or build a dedicated one-step image restoration model.

The project implements native model families and connects them to shared training workflows for parameter selection, optimizer updates, previews, and saving. Families share a configuration structure while retaining their own text encoding, VAE, time sampling, positional encoding, and conditioning rules. Moving to another family keeps the training workflow familiar, with assets and data specified by its recipe.

Training runs through the command line, with reusable recipes stored in TOML. Ordinary generative fine-tuning encodes data first and reads latents and text features from caches; restoration uses its own image and VAE path. Inference weights and resume state are saved separately so results can be used for generation while retaining everything needed to continue training.

## What you can do

| Capability | Workflow |
| --- | --- |
| Generative fine-tuning | Image, video, and audio-video flow matching with adapters or full parameters |
| One-step restoration | Paired LQ/HQ or synthetic degradation, time calibration, VAE encoder adaptation, and multiple losses |
| Multiple datasets | Named groups, shared defaults, prompts, repeats, resolutions, and aspect-ratio buckets |
| Memory management | Block recomputation, attention activation offload, H2D frozen-weight caching, VBAR, and quantized bases |
| Monitoring and resume | Fixed-condition previews, JSONL, TensorBoard, separate resume states, and retention controls |
| Model use | Image/video generation, restoration, adapter publication, and weight precision conversion |

## Supported models

| Task | Families |
| --- | --- |
| Image | Z-Image · FLUX · FLUX.2 · FLUX3 · MOSS Image · Qwen Image · Qwen Image 2.1 · Krea2 |
| Video | Wan 2.1 / 2.2 |
| Audio-video | MiniMaxH3 |

All ten families support flow-matching training. See [models and variants](docs/en/training/models.md) for restoration support, editing/layer conditions, Base/Turbo/distilled variants, and required assets. Match the family, variant, and actual weights when selecting a recipe.

## From data to a model

```text
Model assets + images/video/audio-video + text and conditions
                         │
                      validate    Check configuration and combinations
                         │
                       cache      Encode latents, text, and conditions
                         │
                       train      Forward, backward, accumulation, updates
                         ├── previews/   Observe fixed-condition outputs
                         ├── logs/       Losses, learning rates, resources
                         ├── resume/     Continue the same training run
                         └── weights/    Generate, restore, or load externally
```

Named dataset groups independently specify sources, cache directories, resolutions, batch sizes, and repeats, with shared settings in `dataset.defaults`. Image data supports caption sidecars, shared prompts, aspect-ratio buckets, and model-specific references, masks, or layers. Video and audio-video follow their own frame-count, frame-rate, audio, and condition rules.

Encoded caches avoid rerunning VAE and text encoders on every ordinary flow-matching update. Cache identities include source contents, encoding assets, and preprocessing; rebuild after changing those inputs. Restoration still processes images and its VAE during training. See [datasets](docs/en/training/datasets.md).

## Memory, precision, and execution

Model size, training dimensions, trainable parameters, and loss networks jointly determine resource use. UniTuner exposes separate policies for different resources:

| Policy | Purpose | Requirements |
| --- | --- | --- |
| Resident | Keep weights on the compute device | Default; supports adapters and full training |
| Block recomputation | Reduce saved forward activations | Replay native block computation during backward |
| Attention activation offload | Stage attention's saved activations on the host | CUDA; additional RAM and transfers |
| H2D frozen-block cache | Load frozen weights on demand and reuse resident blocks | CUDA adapters; automatic capacity calibration within this mode |
| VBAR | Manage frozen-weight residency through virtual addresses and physical pages | CUDA adapters and the `vbar` extra |
| Quantized frozen base | Reduce frozen-weight storage | Supported quantized layouts and adapter training |
| CUDA compilation | Compile training block computation with Inductor | Opt-in; initial compilation overhead |

Compute precision supports FP32, CUDA BF16, and FP16; `auto` resolves it from base-model files. Trainable master parameters stay FP32, and FP16 uses fixed loss scaling. Text encoding and export precision have separate settings. Residency is explicit: resident mode does not automatically switch to offload when memory runs out.

Start with default eager/resident execution, then select [execution policies](docs/en/training/execution.md), [precision](docs/en/training/precision.md), and [quantized bases](docs/en/training/quantized-base.md) as needed.

## Quick start

Install Python **3.12+**, [uv](https://docs.astral.sh/uv/), and obtain local model assets. CUDA training requires an NVIDIA driver compatible with the project's Torch version. See [installation](docs/en/installation.md) for optional dependencies and environment checks.

```bash
git clone https://github.com/nukuihayu/UniTuner.git
cd UniTuner
uv python install 3.13
uv sync --locked --no-dev --extra media --extra text
cp docs/examples/tutorial-lora.toml train.toml
```

Edit the model, VAE, text encoder, and dataset paths in `train.toml`, then run:

```bash
uv run --no-dev unituner validate train.toml
uv run --no-dev unituner cache train.toml --device cuda
uv run --no-dev unituner train train.toml --device cuda
```

> [!TIP]
> This starts a Z-Image Base LoRA run. Follow [your first LoRA](docs/en/getting-started.md) for dataset layout, settings, previews, and image generation. Use the matching [example](docs/en/examples/README.md) for other families.

### Keep configuration focused

Edit fields in their existing tables. CLI settings override TOML:

```bash
uv run --no-dev unituner train train.toml --device cuda \
  --train.learning-rate 5e-5 --train.max-updates 2000
```

| Configuration table | Responsibility |
| --- | --- |
| `[model]` | Family, variant, local weights, and encoding assets |
| `[dataset.groups.NAME]` | Source, cache, dimensions, batch size, and repeats |
| `[train]`, `[train.adapter]` | Parameter mode, algorithm, rank, learning rate, and update count |
| `[method]` | Flow matching or restoration, plus optional objectives |
| `[execution]` | Precision, recomputation, weight residency, and compilation |
| `[preview]`, `[output]` | Observation settings, saving, logs, and retention |

Read [TOML and CLI configuration](docs/en/configuration.md) for syntax and [parameter reference](docs/en/training/parameters.md) for defaults and constraints.

## Training outputs

```text
runs/<run-name>/
├── weights/     # Inference models, adapters, and numbered exports
├── resume/      # Model/method, optimizer, RNG, and data cursor state
├── previews/    # Generated or restored samples
└── logs/        # Complete logs and training metrics
```

Use files in `weights/` for inference and states in `resume/` to continue training:

```bash
uv run --no-dev unituner train train.toml --device cuda \
  --resume runs/garden/resume/garden.state-00000050.safetensors
```

> [!IMPORTANT]
> `max_updates` is the target total update count. An adapter file does not contain full resume state. Each `cache` invocation re-encodes inputs and replaces the old cache after success; do not train against that cache during rebuilding.

## Preview, publish, and use the result

Previews run at configured update intervals with fixed prompts, seeds, and conditions. JSONL retains per-update metrics, and optional TensorBoard events expose losses, gradient norms, learning rates, and memory use. Saving and preview cadences are independent, so observation frequency can be adjusted without changing the objective.

Published adapters use the selected family's consumer layout, including required projection mappings. Load them with the matching model and supported adapter loader in ComfyUI, or use UniTuner's `generate`, `generate-video`, and `restore` commands as appropriate. Base weights, VAE, text assets, conditions, and sampling conventions must agree when comparing outputs across tools.

Weight tools also import supported external layouts, convert floating precision, and handle supported quantized formats. Sensitive-layer safeguards are model-specific. See [inference](docs/en/inference.md) and [weight conversion](docs/en/weight-conversion.md) for formats and command examples.

## Find your workflow

| I want to… | Read |
| --- | --- |
| Train my first model | [Install](docs/en/installation.md) → [LoRA tutorial](docs/en/getting-started.md) |
| Configure training and data | [Training guide](docs/en/training/README.md) · [Datasets](docs/en/training/datasets.md) |
| Train restoration | [Restoration tutorial](docs/en/restoration-tutorial.md) · [Losses and time](docs/en/training/restoration.md) |
| Train video or audio-video | [Wan / MiniMaxH3](docs/en/training/video.md) |
| Fit training in GPU memory | [Execution](docs/en/training/execution.md) · [Quantized bases](docs/en/training/quantized-base.md) |
| Use trained weights | [Inference and ComfyUI](docs/en/inference.md) · [Weight conversion](docs/en/weight-conversion.md) |
| Diagnose errors or unexpected results | [Troubleshooting](docs/en/troubleshooting.md) · [Logging](docs/en/logging.md) |

For development and model integration, read [architecture](docs/en/architecture.md). Documentation is available in [English](docs/en/README.md) and [Chinese](docs/README.md), sharing executable TOML examples.
