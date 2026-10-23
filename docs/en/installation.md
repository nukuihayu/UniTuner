# Installation and environment

[简体中文](../installation.md) · [Documentation](README.md)

Prepare the environment here, then follow [your first training run](getting-started.md) or choose a [model-specific example](examples/README.md).

<details>
<summary>On this page</summary>

- [Requirements](#requirements)
- [Install](#install)
- [Optional dependencies](#optional-dependencies)
- [Check CUDA](#check-cuda)
- [Tokenizers and offline use](#tokenizers-and-offline-use)
- [Mirrors and fixed environments](#mirrors-and-fixed-environments)
- [Development](#development)

</details>

## Requirements

| Component | Requirement and purpose |
| --- | --- |
| Python | 3.12 or newer; `uv python install 3.13` installs a suitable interpreter |
| Package management | Use uv and the repository's `uv.lock` |
| PyTorch | Torch 2.13 series; the lockfile determines the resolved build |
| CUDA | NVIDIA GPU with a matching driver; required for BF16/FP16 training, compilation, and frozen-block offload |
| CPU | Supports applicable FP32 paths, without CUDA-only execution options |
| Model files | Supply the family's denoiser, VAE, text encoder, and required auxiliary assets |
| Disk and host memory | Weights, encoded caches, outputs, and resume states; offload uses RAM, and cache rebuilding retains the old cache until publication |

GPU memory requirements depend on the model, dimensions, batch, parameter mode, and losses. Start with a small batch and consult [execution policies](training/execution.md).

## Install

```bash
git clone https://github.com/nukuihayu/UniTunner.git
cd UniTunner
uv python install 3.13
uv sync --locked --no-dev --extra media --extra text
uv run --no-dev unitunner --help
```

Commands assume the repository root as the working directory. Installing dependencies does not download model weights.

## Optional dependencies

| Extra | Provides | Required for |
| --- | --- | --- |
| `media` | Image, video, and audio IO | Cache production, previews, and generation |
| `text` | Text models, tokenizers, and processors | Text cache encoding, conditioned previews, and inference |
| `kitchen` | comfy-kitchen quantization and optional operators | Applicable conversion, quantization, and inference attention paths |
| `vbar` | comfy-aimdo memory page management | `execution.residency = "vbar"` |

To include quantization and VBAR:

```bash
uv sync --locked --no-dev --extra media --extra text --extra kitchen --extra vbar
```

Keep listing required extras when syncing again. Training from existing caches without previews does not need to load a text encoder. Restoration still needs its VAE and image IO.

## Check CUDA

```bash
uv run --no-sync --no-dev python -c "import torch; print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

If CUDA is unavailable, check the driver, active Python environment, and Torch build. `--device cuda` selects a device; it does not install drivers or replace Torch. For CPU workflows explicitly choose `--device cpu --execution.precision fp32`. Automatic precision follows the weights and does not guarantee a CPU-compatible selection.

The `unitunner train` entry initializes VBAR before Torch. If a Notebook process already imported Torch, launch a fresh CLI process for VBAR.

## Tokenizers and offline use

Text encoder weights and tokenizers are separate assets. Default sources may download tokenizer files, but not encoder weights. For offline operation:

```toml
[model.text.tokenizer]
source = "/models/tokenizer"
offline = true
```

Use `subfolder = "tokenizer"` if the files are nested below that directory. FLUX CLIP tokenizers and Qwen processors have separate settings; see [text encoding](training/text-encoding.md).

## Mirrors and fixed environments

Set a PyPI mirror when needed:

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --locked --no-dev --extra media --extra text
```

This controls Python package sources, not model or interpreter downloads. After installation, `uv run --no-sync --no-dev unitunner …` avoids synchronizing dependencies at launch.

## Development

```bash
uv sync --locked --group dev --extra media --extra text
make check
make test
```

`make check` runs Ruff, Ty, and Pyright; `make test` runs project tests. Some tests require CUDA, optional dependencies, or local assets. See [architecture](architecture.md) for model integration and resource ownership.

---

[Documentation](README.md) · [Parameter reference](training/parameters.md) · [Troubleshooting](troubleshooting.md)
