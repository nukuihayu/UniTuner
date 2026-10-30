# Configuration examples

[简体中文](../../examples/README.md) · **English**

[Documentation](../README.md) · [Training](../training/README.md) · [Models](../training/models.md)

Both language editions use the configurations in `docs/examples/`. Copy a configuration and fill in your model and dataset paths. Relative paths resolve from the command's working directory.

<details>
<summary>On this page</summary>

- [How to use an example](#how-to-use-an-example)
- [Complete tutorial pairs](#complete-tutorial-pairs)
- [Training and caching](#training-and-caching)
- [Image restoration](#image-restoration)
- [Inference](#inference)

</details>

## How to use an example

1. Copy the file to your run folder; change model, data, cache and output paths first.
2. Keep the family/variant/version consistent with the actual weights. Read the bilingual comments beside each field.
3. Active values form the example recipe. Commented values are optional: do not enable every option at once. Defaults such as 60-second summaries and retention limits normally need no entry.
4. Training: validate, then cache, then train. Inference: use generate, generate-video or restore as stated in the file header.
5. For another experiment use a new output directory/name; use a resume state only to continue the same run.

See [dataset filling](../training/datasets.md), [trainer workflow](../training/README.md) and [all parameters](../training/parameters.md). Both language editions share the same TOML files.

## Complete tutorial pairs

Use each training and inference pair together, with matching asset paths, output names, and inference settings.

| Workflow | Training | Inference | Walkthrough |
| --- | --- | --- | --- |
| Z-Image style LoRA | [tutorial-lora.toml](../../examples/tutorial-lora.toml) | [tutorial-generate.toml](../../examples/tutorial-generate.toml) | [First training run](../getting-started.md) |
| Z-Image one-step restoration | [tutorial-restoration.toml](../../examples/tutorial-restoration.toml) | [tutorial-restore.toml](../../examples/tutorial-restore.toml) | [Restoration tutorial](../restoration-tutorial.md) |

## Training and caching

Training examples use the training schema; cache-only examples use the encoding schema. Usually run `unituner cache CONFIG --device cuda`, then `unituner train CONFIG --device cuda`. Training-only examples with just a denoiser require caches built using the corresponding complete cache configuration.

| Task | Example | Notes |
| --- | --- | --- |
| Multiple datasets | [data-groups.toml](../../examples/data-groups.toml) | Shared defaults, independent caches, group repeats |
| Image caching only | [image-cache.toml](../../examples/image-cache.toml) | Encoder assets and data only |
| VAE calibration only | [mid-timestep.toml](../../examples/mid-timestep.toml) | Paired images and VAE, no denoiser/cache |
| Z-Image adapter | [flow.toml](../../examples/flow.toml) | Default LoRA; switch algorithm to loha/lokr |
| Z-Image full | [full.toml](../../examples/full.toml) | FP32, no `train.adapter` |
| Image previews | [flow-preview.toml](../../examples/flow-preview.toml) | Prompt-based generation during training |
| LayerSync + TREAD + REPA | [flow-features.toml](../../examples/flow-features.toml) | Replace zero SHA256 with the encoder's actual digest; methods are independently removable |
| FLUX Dev | [flux-flow.toml](../../examples/flux-flow.toml) | T5 path and bundle-local CLIP |
| FLUX Fill | [flux-fill-flow.toml](../../examples/flux-fill-flow.toml) | One reference directory plus mask; for Kontext change weights/variant and remove `mask_source` |
| FLUX.2 | [flux2-flow.toml](../../examples/flux2-flow.toml) | Klein 4B base without references |
| FLUX.2 references | [flux2-reference-flow.toml](../../examples/flux2-reference-flow.toml) | Ordered references use their own aspect buckets; equal-sized pairs stay aligned |
| FLUX3 | [flux3-flow.toml](../../examples/flux3-flow.toml) | See model guide for references |
| MOSS | [moss-flow.toml](../../examples/moss-flow.toml) | Native skip-layer preview parameters must match topology |
| MOSS Edit | [moss-edit-flow.toml](../../examples/moss-edit-flow.toml) | Reference-conditioned editing; text-start layout |
| Qwen original | [qwen-image-flow.toml](../../examples/qwen-image-flow.toml) | Text-to-image |
| Qwen edit-2511 | [qwen-image-edit-flow.toml](../../examples/qwen-image-edit-flow.toml) | Target/reference file stems must match |
| Qwen layered | [qwen-image-layered-flow.toml](../../examples/qwen-image-layered-flow.toml) | RGBA foreground layers and composite |
| Qwen Image 2.1 | [qwen-image-2-1-flow.toml](../../examples/qwen-image-2-1-flow.toml) | Separate 64-channel VAE and Qwen3-VL |
| Krea2 | [krea2-flow.toml](../../examples/krea2-flow.toml) | RAW/Turbo must match weights |
| Wan caching | [wan-cache.toml](../../examples/wan-cache.toml), [single-frame cache](../../examples/wan-single-frame-cache.toml) | Video and independent condition-frame caches |
| Wan training | [wan-flow.toml](../../examples/wan-flow.toml), [with previews](../../examples/wan-flow-preview.toml) | Native video training |
| H3 caching | [minimax-h3-cache.toml](../../examples/minimax-h3-cache.toml) | Native AV JSONL; full AV VAE assets |
| H3 training | [minimax-h3-train.toml](../../examples/minimax-h3-train.toml), [with previews](../../examples/minimax-h3-train-preview.toml) | Existing caches, optional AV previews |

For external split H3 caches, use [minimax-h3-cache-import.json](../../examples/minimax-h3-cache-import.json):

```bash
uv run unituner caches import SPECIFICATION DESTINATION
```

This is an explicit conversion specification, not a training TOML.

## Image restoration

| Task | Example | Dependencies and limits |
| --- | --- | --- |
| Z-Image default losses | [restoration.toml](../../examples/restoration.toml) | DINOv3-ConvNeXt and VGG19 weights |
| Z-Image Turbo skin restoration | [oil-skin-lora.toml](../../examples/oil-skin-lora.toml) | Paired real images, DINO/VGG/GAN objectives; replace local paths |
| Z-Image full | [restoration-full.toml](../../examples/restoration-full.toml) | Same losses, full denoiser, block recomputation |
| Synthetic degradation | [restoration-synthetic.toml](../../examples/restoration-synthetic.toml) | No paired LQ; degradation belongs in dataset |
| Mixed sources | [restoration-mixed.toml](../../examples/restoration-mixed.toml) | Real pairs and synthetic degradation; per-source caches/repeats |
| SNR calibration | [restoration-calibration.toml](../../examples/restoration-calibration.toml) | Standalone/pre-training; LRR + L1, no perceptual weights |
| FLUX / FLUX.2 / FLUX3 | [FLUX](../../examples/flux-restoration.toml), [FLUX.2](../../examples/flux2-restoration.toml), [FLUX3](../../examples/flux3-restoration.toml) | LRR + L1 |
| MOSS | [moss-image-restoration.toml](../../examples/moss-image-restoration.toml) | LRR + L1 |
| Qwen / Qwen 2.1 | [Qwen](../../examples/qwen-image-restoration.toml), [Qwen 2.1](../../examples/qwen-image-2-1-restoration.toml) | Qwen original only |
| Krea2 | [krea2-restoration.toml](../../examples/krea2-restoration.toml) | Native training posterior |

After replacing paths:

```bash
uv run unituner mid-timestep docs/examples/restoration-calibration.toml --device cuda
uv run unituner cache docs/examples/restoration-calibration.toml --device cuda
uv run unituner train docs/examples/restoration-calibration.toml --device cuda
```

The first command needs only VAE/images and no cache. It prints JSON without rewriting TOML. This example sets `calibrate=true`, so training calibrates automatically. To freeze a result, disable calibration and copy the output sigma into the configuration. Default `midtimestep=244` does not mean equal sigma for every family; see [restoration](../training/restoration.md).

## Inference

Inference configurations use separate schemas and cannot be passed to `unituner validate`. Each inference command loads and validates its own configuration before reading local models.

| Command | Examples |
| --- | --- |
| `unituner generate CONFIG --device cuda` | [Z-Image](../../examples/generate.toml), [FLUX](../../examples/flux-generate.toml), [FLUX.2](../../examples/flux2-generate.toml), [FLUX3](../../examples/flux3-generate.toml), [MOSS](../../examples/moss-generate.toml), [MOSS Edit](../../examples/moss-edit-generate.toml), [Qwen](../../examples/qwen-image-generate.toml), [Qwen layered](../../examples/qwen-image-layered-generate.toml), [Krea2](../../examples/krea2-generate.toml) |
| `unituner generate-video CONFIG --device cuda` | [Wan](../../examples/wan-generate-video.toml), [H3](../../examples/minimax-h3-generate.toml) |
| `unituner restore CONFIG --device cuda` | [Z-Image](../../examples/restore.toml), [Qwen](../../examples/qwen-image-restore.toml), [Krea2](../../examples/krea2-restore.toml) |

Adapter paths and algorithm/rank/alpha must match training publications. For full models, replace `model.weights`. Restoration `encoder_weights` points to the exported encoder adapter and uses the sigma recorded by the training publications.

---

[Documentation](../README.md) · [Parameter reference](../training/parameters.md) · [Troubleshooting](../troubleshooting.md)
