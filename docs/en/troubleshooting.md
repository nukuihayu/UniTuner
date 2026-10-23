# Troubleshooting

[简体中文](../troubleshooting.md) · [Documentation](README.md)

Locate the failing stage: validation → cache encoding → training → preview → export/inference. Keep the command, effective configuration, and surrounding logs to distinguish asset, data, and execution issues.

<details>
<summary>On this page</summary>

- [Configuration and assets](#configuration-and-assets)
- [Caches and datasets](#caches-and-datasets)
- [Out of GPU memory](#out-of-gpu-memory)
- [Numerical issues and training quality](#numerical-issues-and-training-quality)
- [Resume and output](#resume-and-output)
- [ComfyUI and standalone outputs differ](#comfyui-and-standalone-outputs-differ)
- [Information for a useful issue report](#information-for-a-useful-issue-report)

</details>

## Configuration and assets

| Symptom | Check and action |
| --- | --- |
| Unknown / extra field | Check the owning table, [parameters](training/parameters.md), and command `--help`; generation uses its own schema |
| Existing path cannot be found | Relative paths use the working directory; check shard completeness and distinguish tokenizers from encoder weights |
| Key / shape / topology mismatch | Verify family, variant, version, component, and format; renaming a model does not convert its architecture |
| Tokenizer download or offline failure | Set a local tokenizer source and, when needed, subfolder; verify separate processor and CLIP sources |
| Missing optional dependency | Install the corresponding [extra](installation.md) into the environment running the command |
| CUDA unavailable | Check driver, CUDA-enabled Torch, and visible devices; CPU workflows need FP32 without CUDA-only policies |

## Caches and datasets

### Images or captions do not take effect

Images are discovered directly inside the source directory. Captions use matching UTF-8 sidecar names. A group/default `prompt` overrides sidecars, including `prompt = ""`. Rebuild after changing media, conditions, VAE, text settings, or crop policy.

### Cache digest, layout, or identity mismatch

Do not edit safetensors or manifests manually. Select the cache for the correct model and group, then rebuild it. Groups require independent cache directories; references, masks, and HQ/LQ images must follow [pairing rules](training/datasets.md).

Every `cache` invocation rebuilds and replaces directories after success. Old and staged caches coexist on disk during rebuilding; do not train against the same directory. If rollback fails, preserve the backup path reported in logs before investigating permissions or disk problems.

## Out of GPU memory

| Stage | Start here |
| --- | --- |
| Base model loading | For adapter training consider frozen-block offload, VBAR, or a supported quantized base; inspect other GPU processes |
| Forward or backward | Reduce batch/dimensions; enable block recomputation; consider attention activation offload |
| First larger bucket | Automatic capacity cannot guarantee unseen dimensions will fit; reduce the largest area |
| Preview or saving | Previews need additional VAE/text assets; reduce dimensions/sample count and check host RAM |

Resident mode does not automatically switch to offload after OOM. H2D adjusts its cache within selected `frozen-blocks` mode; VBAR manages pages within selected VBAR mode. Accumulation increases effective batch size without reducing one microbatch's activations. See [execution](training/execution.md).

## Numerical issues and training quality

### NaN / Inf loss or gradients

Inspect cached inputs, learning rate, loss weights, and precision. For FP16, reduce `execution.fp16_loss_scale`, or use BF16 on supported hardware. Nonfinite gradients fail before updating parameters rather than silently skipping a batch. Do not bypass resume identity checks after changing numerical policies.

### Loss improves but previews do not

Fix preview prompts, seeds, dimensions, and sampling settings. Inspect captions, trigger words, layer selection, dataset proportions, and update count. Check component losses for a dominating objective. Base, Turbo, and distilled variants do not share universal steps/guidance settings.

### Training works but previews fail

Training can read encoded latents while previews require VAE, text weights, tokenizers, and conditions. Verify those assets and the preview variant. Disabling previews does not fix a mismatched generation setup.

## Resume and output

- Resume from `.state-…safetensors` in `resume/`, not an adapter in `weights/`.
- `max_updates` is the target total; a completed state performs no additional updates.
- Frozen-reference states require identical frozen base weights, including their quantized contents.
- Omitting `output.directory` creates a fresh run directory. Fresh training in an explicit directory clears recognized same-name artifacts; use `--resume` to continue.
- Defaults retain one resume state and all exported weights. Check retention settings when older states are missing.

## ComfyUI and standalone outputs differ

Check in this order:

1. Load a file from `weights/` with the matching base family, variant/version, and adapter algorithm.
2. Verify consumer support for the exported layout. Fused projections may require splitting; renaming keys alone is insufficient.
3. Match VAE, text encoder, positive/negative prompts, conditions, and reference order.
4. Match sampler, time/shift convention, steps, guidance, dimensions, and adapter strength.
5. Check quantization, compute precision, and attention backend. Identical seeds across runtimes need not produce identical noise.

Training previews use native evaluation. Enabling training compilation does not require copying its policy into external inference. See [weight conversion](weight-conversion.md) and [inference](inference.md).

## Information for a useful issue report

Include the command, model family/variant, redacted configuration, full exception, Torch/CUDA versions, and the failing stage. See [logging](logging.md) for log locations and reading metrics.

---

[Documentation](README.md) · [Parameter reference](training/parameters.md) · [Troubleshooting](troubleshooting.md)
