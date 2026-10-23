<p align="center"><img src="../assets/unitunner-logo.png" alt="UniTunner" width="560"></p>

# Documentation

[简体中文](../README.md) · [Project home](../../README.md)

## Start here

| Page | Contents |
| --- | --- |
| [Installation and environment](installation.md) | Dependencies, CUDA, optional extras, offline assets |
| [Your first LoRA](getting-started.md) | Prepare images, cache, train, preview, and generate |
| [TOML and CLI](configuration.md) | Configuration ownership, overrides, and command reference |
| [Training concepts](concepts.md) | Latents, flow matching, adapters, and update counts |

## Training workflows

| Page | Contents |
| --- | --- |
| [Training guide](training/README.md) | Parameter modes, optimization, previews, and resume |
| [Datasets](training/datasets.md) | Groups, captions, buckets, references, and pairing |
| [Restoration tutorial](restoration-tutorial.md) | Paired LQ/HQ training from start to inference |
| [Video and audio-video](training/video.md) | Wan and MiniMaxH3 data, assets, and commands |
| [Configuration recipes](examples/README.md) | Shared TOML recipes organized by task and model |

## Configuration reference

| Page | Contents |
| --- | --- |
| [All parameters](training/parameters.md) | Types, defaults, constraints, and table locations |
| [Models and assets](training/models.md) | Families, variants, conditioning, and components |
| [Memory and execution](training/execution.md) | Residency, H2D, VBAR, recomputation, and compilation |
| [Precision](training/precision.md) | Storage, computation, AMP, and export |
| [Quantized bases](training/quantized-base.md) | Adapter training with frozen quantized weights |
| [Text encoding](training/text-encoding.md) | Text weights, tokenizers, precision, and quantization |
| [Auxiliary objectives](training/objectives.md) | LayerSync, REPA, TREAD, and wavelet loss |
| [Restoration reference](training/restoration.md) | Time calibration, losses, and encoder adaptation |

## Use and diagnose

| Page | Contents |
| --- | --- |
| [Inference](inference.md) | Use published adapters and full models |
| [Weight conversion](weight-conversion.md) | Formats, consumer layouts, and precision |
| [Weight analysis](weight-analysis.md) | Inspect trained weights and adapter changes |
| [Logs and TensorBoard](logging.md) | Terminal, JSONL, and event metrics |
| [Troubleshooting](troubleshooting.md) | Assets, cache, OOM, resume, and output differences |

## Project reference

| Page | Contents |
| --- | --- |
| [Architecture](architecture.md) | Module boundaries and resource ownership |
| [Execution internals](training/execution-internals.md) | Weight lifetimes and training execution boundaries |
