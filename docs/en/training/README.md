# Trainer tutorial

[简体中文](../../training/README.md) · [First run](../getting-started.md) · [Parameter guide](parameters.md)

Follow this page for image training. For Wan and MiniMax-H3 data preparation, caching and previews, see the [video training tutorial](video.md).

<details>
<summary>On this page</summary>

- [Choose a method](#choose-a-method)
- [Flow matching: data to training](#flow-matching-data-to-training)
- [Restoration and intermediate time](#restoration-and-intermediate-time)
- [Select trainable layers](#select-trainable-layers)
- [How parameters interact](#how-parameters-interact)
- [Shared training parameters](#shared-training-parameters)
- [Execution policies](#execution-policies)
- [Previews](#previews)
- [Saving and resuming](#saving-and-resuming)
- [Multiple datasets](#multiple-datasets)
- [Independent commands, shared captions and cuDNN](#independent-commands-shared-captions-and-cudnn)
- [Inference and publication](#inference-and-publication)
- [Common configuration problems](#common-configuration-problems)

</details>

## Choose a method

Choose the objective separately from which parameters are updated:

| Choice | Configuration | Purpose |
| --- | --- | --- |
| Image/video generation tuning | method.kind=flow-matching | Subjects, style, editing and video conditioning |
| One-step image restoration | method.kind=restoration | Aligned LQ/HQ or explicitly synthesized LQ |
| Adapter | train.parameter_mode=adapter plus train.adapter | Export LoRA/LoHa/LoKr |
| Full model | train.parameter_mode=full, remove train.adapter | Export a trained base; greater memory and disk use |

A restoration LoRA is restoration + adapter. See [model support](models.md) before selecting a recipe.

## Flow matching: data to training

1. Choose the exact family/variant [example](../examples/README.md) and replace paths.
2. Fill model, dataset.groups.NAME, train, method, execution and output in one TOML.
3. Add previews with fixed prompts/seeds when you need visual comparisons.
4. Validate, cache and train in order.

```bash
uv run --no-dev unitunner validate train.toml
uv run --no-dev unitunner cache train.toml --device cuda
uv run --no-dev unitunner train train.toml --device cuda
```

cache validates only encoding inputs; it does not require optimizer, adapter or training-loss settings. Copy image-cache, wan-cache or minimax-h3-cache for standalone caching. Cached image flow training without previews need not load the VAE/text encoder. Restoration still reads source images and runs the VAE during training.

## Restoration and intermediate time

Copy [tutorial-restoration](../../examples/tutorial-restoration.toml). Point source at HQ and low_quality_source at stem-matched LQ. Remove synthetic-degradation tables when supplying real pairs. For fixed time set midtimestep with calibrate=false. For automatic time estimation use calibrate=true without explicit sigma. Follow the [restoration tutorial](../restoration-tutorial.md).

## Select trainable layers

```toml
[train]
include_layers = ["layers.*"]
exclude_layers = ["layers.0.*"]
```

These are Z-Image native names. Check another family's module names before reusing the patterns. Patterns refer to native layers/modules/parameters, not exported ComfyUI LoRA keys. Empty include_layers selects all eligible targets; exclusions apply afterward. Start unrestricted, then test selected layers.

In full mode this selects trainable base parameters. In adapter mode it limits adapter targets. Correct patterns that select no eligible targets rather than substituting unrelated export prefixes.

## How parameters interact

### Effective updates and batches

For one group with full batches, effective batch is approximately batch_size × accumulation_steps. Tail batches may be smaller, and groups may have different batch sizes. Gradients are normalized by consumed samples. update advances only after a successful optimizer update. Restoration generator/discriminator phases share one logical update.

Increasing accumulation from 1 to 4 makes each update contain four microbatches. It does not provide four times the training for free. Save cadence, previews, warmup and max_updates all count effective updates.

### Learning rates and warmup

```toml
[train]
learning_rate = 5e-5
weight_decay = 1e-4
max_updates = 10000
accumulation_steps = 4
max_grad_norm = 1.0
lr_schedule = "cosine"
warmup_updates = 100

[train.optimizer]
betas = [0.9, 0.999]
eps = 1e-8
backend = "auto"
```

learning_rate sets the base update magnitude. warmup_updates ramps it linearly and must be smaller than max_updates. constant stays constant after warmup; linear/cosine decay across the configured horizon. Start with default decay/clipping and inspect component losses and gradient norms when diagnosing instability.

### LoRA, LoHa and LoKr

```toml
[train.adapter]
algorithm = "lora"
rank = 32
alpha = 32.0
```

Ordinary LoRA scales by alpha/rank. Start with equal alpha/rank and compare capacity while holding other settings fixed. Switching algorithms should start a new run rather than reuse old optimizer state. Inference must match the exported algorithm/rank/alpha.

## Shared training parameters

| Priority | Fields to understand | Guidance |
| --- | --- | --- |
| Paths | model weights/VAE/text, source, cache_directory | Existing absolute paths simplify troubleshooting |
| Task | family/variant/version, method.kind, parameter_mode | Match actual weights and objective |
| Data | resolution, batch_size, repeats | Dimensions affect activations; repeats changes group frequency |
| Optimization | learning_rate, max_updates, accumulation_steps | Validate with a short run before committing a larger budget |
| Capacity | algorithm, rank, alpha | Compare while holding other variables fixed |
| Monitoring | preview, tensorboard | Fixed preview conditions plus total and component losses |
| Outputs | name, save_every_updates | Increase interval if saves are expensive |

The [parameter guide](parameters.md) lists defaults, bounds and choices. Default values usually need not appear in your TOML; expanded examples expose them for learning and tuning.

## Execution policies

Choose compute precision explicitly when needed: bf16 for CUDA BF16 recipes, fp32 for CPU checks. auto follows checkpoint dtype; a quantized base can resolve to BF16.

For memory pressure compare smaller batches, block recomputation, frozen-weight offload or quantized adapter bases. Full training requires floating bases. Compilation and cuDNN benchmarking add startup costs: compare stable iterations, not the first step. See [execution](execution.md).

## Previews

Image flow uses preview plus preview.samples array tables. Restoration uses preview.num_images to rotate LQ/restored/HQ comparisons from actual data; do not add generation samples. Wan/H3 use preview.videos and preview.audio_video; copy their preview examples.

every_updates counts effective updates, including a final preview. at_start only renders a fresh-run baseline. Previews preserve training RNG and the data cursor.

Preview and save intervals can differ: preview every 100 updates and save every 1000. If previews are slow, reduce their count/frequency before changing training batch settings.

## Saving and resuming

```toml
[output]
name = "portrait-lora"
save_every_updates = 1000
tensorboard = "runs/tensorboard"
# directory = "runs/portrait-lora"
# max_resume_checkpoints = 1
# max_weight_checkpoints = 0
# save_precision = "fp32"
```

Omitting directory creates runs/portrait-lora_<timestamp>/. Explicit directories are reused: a fresh run replaces recognized same-name artifacts; --resume preserves and continues training state.

| Location | Contents | Use |
| --- | --- | --- |
| weights/NAME-00001000.safetensors | Update-1000 adapter or base | Compare versions, infer or distribute |
| weights/NAME.safetensors | Latest export | Stable inference path updated on every save |
| weights/NAME-00001000.encoder.safetensors | Optional restoration encoder adapter | Pair with the same update's weights |
| resume/NAME.state-00001000.safetensors | Model/method, optimizer, RNG and data cursor | --resume only |
| logs/, previews/ | Full metrics and visual samples | Troubleshooting and version selection |

Weights and independent resume state save at the interval and final update. Defaults keep one resume snapshot and all weight versions. Positive max_* values keep the newest N snapshots/sets; 0 is unlimited. Encoders follow their weight set. Old files are removed only after a complete successful save.

```bash
uv run --no-dev unitunner train train.toml --device cuda --resume runs/portrait-lora/resume/portrait-lora.state-00001000.safetensors
```

Resume continues to total max_updates. Preserve the base and data; numerical settings undergo identity checks. save_precision affects inference exports only. checkpoint_storage=frozen-reference can reduce duplicated frozen data but requires the identical frozen base on restore.

## Multiple datasets

Put shared resolution/batch/prompt fields in dataset.defaults. Each dataset.groups.NAME overrides its own paths, repeats or prompt. Use independent cache directories. Names use ASCII letters/digits/underscores and cannot start with digits. Restoration groups must share one calibration resolution.

See [dataset configuration](datasets.md) and [data-groups.toml](../../examples/data-groups.toml) for inheritance and relative frequencies.

## Independent commands, shared captions and cuDNN

cache requires encoding assets/data. mid-timestep requires only a VAE, paired images and calibration controls. generate/restore use inference configurations. None reuse full training validation. TOML is optional when supplying CLI fields; consult --help and use --set for nested keys.

Shared captions belong in dataset.defaults.prompt or a group's prompt. Runtime cuDNN/TF32 options belong in runtime and can be set independently for cache/train/inference. CUDA convolutions default to cudnn_benchmark=true; for repeatability use cudnn_deterministic=true and normally disable benchmarking. See [execution](execution.md).

### TensorBoard

Set only output.tensorboard to an event root. With no fixed output directory, the run uses training name/timestamp. Explicit directories use a fixed run name. Resume continues the run; no separate TensorBoard name is needed.

```bash
tensorboard --logdir runs/tensorboard --host 0.0.0.0 --port 6006
```

Inspect loss/main or loss/generator, restoration components, discriminator loss, gradient norms, learning rates, update sample counts, timing and memory. Total loss is weighted; components retain their own scales. Do not compare recipe quality solely by total loss magnitude.

## Inference and publication

Adapters export ComfyUI keys for their matching model. Load them with that base model's LoRA loader. Full exports become the new base. Restoration encoders are separate VAE deltas and must be paired with the same update's adapter in UniTunner restore. See [inference](../inference.md).

## Common configuration problems

| Symptom | First check |
| --- | --- |
| Unknown/extra field | Correct table path; inference configs do not go through validate |
| Cache identity mismatch | Rebuild after changes to images, captions, encoders or preprocessing |
| Weight shape mismatch | Family, variant/version, VAE and text encoder compatibility |
| Missing tokenizer | Point source at actual files; offline mode needs all files cached |
| CUDA OOM | Inspect peak memory; reduce batch/dimensions, try recomputation/offload |
| Loss falls but images deteriorate | Inspect fixed previews/components, lower LR or shorten the run |

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
