# Inference and model use

[简体中文](../inference.md) · English · [Documentation](README.md)

First identify whether the file is an adapter, a full model, or training state. This determines how to load it. Image generation, video generation, and restoration each use their own inference configuration.

<details>
<summary>On this page</summary>

- [Choose the right file](#choose-the-right-file)
- [Generate with an image LoRA](#generate-with-an-image-lora)
- [Use a full model](#use-a-full-model)
- [Restore an image](#restore-an-image)
- [Generate video or audio-video](#generate-video-or-audio-video)
- [Load in ComfyUI](#load-in-comfyui)
- [Check a failing inference run](#check-a-failing-inference-run)

</details>

## Choose the right file

| File | Purpose | Loading |
| --- | --- | --- |
| `<name>.safetensors` from adapter training | Weight update | Retain the base model and load through `[adapter]` |
| `<name>.safetensors` from full training | Complete trained model | Replace `model.weights`; omit adapter configuration |
| `<name>.encoder.safetensors` | Restoration encoder update | Set `encoder_weights` in `restore` |
| `<name>.state-*.safetensors` | Optimizer and complete training state | Use only with `train --resume` |

The extension alone does not identify a file's role. Use the training mode and filename. Adapters must match the training base model, algorithm, rank, and alpha.

For a quantized training base, use the same quantized artifact at inference. If training quantized a floating source automatically, first [convert the base](weight-conversion.md) with matching format, rotation/group parameters, and device settings, then set inference `model.weights` to that file. Inference configurations have no training `execution.quantization` setting. Previews use the training model directly; switching standalone generation to the original floating base can change outputs.

Text encoders can load packed artifacts or quantize before inference through `[model.text]`, independently choosing FP32/FP16/BF16 computation. Retain matching tokenizer, processor, and source namespaces; see [text encoding](training/text-encoding.md).

## Generate with an image LoRA

For the `garden` adapter from [your first training run](getting-started.md), save this complete configuration as `generate.toml`:

```toml
output = "runs/garden/sample.png"

[model]
family = "z-image"
variant = "base"
weights = "models/z-image.safetensors"
vae = "models/z-image-vae.safetensors"
text_encoder = "models/qwen3"

[adapter]
weights = "runs/garden/weights/garden.safetensors"

[adapter.config]
algorithm = "lora"
rank = 32
alpha = 32.0

[sampling]
prompt = "a garden path in the style of gardenstyle, soft morning light"
negative_prompt = ""
width = 1024
height = 1024
steps = 20
guidance = 4.0
seed = 42
```

Alternatively, copy the [configuration file](../examples/tutorial-generate.toml), then run:

```bash
uv run unitunner generate generate.toml --device cuda
```

`seed` fixes initial random noise for comparison under matching settings. `steps` controls sampling iterations; `width` and `height` set output size. Guidance is model-specific: it may mean CFG or learned guidance embedding, and distilled variants may fix or reject it. Start from the family's [generation example](examples/README.md#inference) when changing models.

To compare base and adapted outputs, copy the configuration, remove `[adapter]` / `[adapter.config]`, and choose another output filename. Keep other settings unchanged to isolate the effect of fine-tuning.

## Use a full model

Copy the family's generation configuration and replace `model.weights` with the complete trained model. Retain the matching VAE, text encoder, and condition assets, and remove adapter sections. The generation command stays the same.

Full mode publishes a complete model even when only selected denoiser layers were trained. Do not load that full publication as a LoRA on top of the original base.

## Restore an image

Restoration uses `restore`; see the [restoration tutorial](restoration-tutorial.md) for the full workflow. Set input/output paths and the matching denoiser/encoder publications:

```bash
uv run unitunner restore restore.toml --device cuda
```

Inference reads sigma from trained artifacts by default. Denoiser and encoder must belong to matching results rather than different restoration times. Inference uses posterior means; results trained with encoder adaptation need the corresponding encoder file.

## Generate video or audio-video

Wan and MiniMaxH3 use `generate-video`:

```bash
uv run unitunner generate-video docs/examples/wan-generate-video.toml --device cuda
uv run unitunner generate-video docs/examples/minimax-h3-generate.toml --device cuda
```

Replace local asset/output paths before running. Dimensions, frame counts, audio, and reference conditions follow family-specific rules. Do not copy an image sampling section into these configurations. See [video and audio-video](training/models.md#video-and-audio-video).

## Load in ComfyUI

Current adapter training publishes ComfyUI keys for the matching model by default. Load these as the base model's LoRA/LoHa/LoKr. ComfyUI must support that model and its fused projection mappings. Restoration encoder files belong to UniTunner's restoration pipeline and are not ordinary denoiser LoRAs.

`adapters export-comfyui` converts existing native Wan/H3 adapters. Current training publications generally need no further conversion. See [publication formats](training/README.md#inference-and-publication).

## Check a failing inference run

Do not pass inference TOML to `unitunner validate`, which checks training only. Each inference command loads its own schema. Check family/variant, base path, adapter algorithm/rank, and condition assets before checking dimensions and sampling options.

Torch attention is the default. Optional Kitchen INT8 changes output numerics; see [INT8 inference attention](training/execution.md#int8-inference-attention) before comparing it.

---

[Documentation](README.md) · [Parameter reference](training/parameters.md) · [Troubleshooting](troubleshooting.md)
