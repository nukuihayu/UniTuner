# Train one-step restoration from existing image pairs

[简体中文](../restoration-tutorial.md) · [Trainer tutorial](training/README.md)

Use real LQ/HQ images without synthesizing degradation. Validate the LRR+L1 workflow first, then add DINOv3, GAN or VGG objectives as needed.

<details>
<summary>On this page</summary>

- [1. Prepare pairs](#1-prepare-pairs)
- [2. Fill data and training settings](#2-fill-data-and-training-settings)
- [3. Choose fixed time or calibration](#3-choose-fixed-time-or-calibration)
- [4. Preview, cache and train](#4-preview-cache-and-train)
- [5. Add perceptual, color or adversarial losses](#5-add-perceptual-color-or-adversarial-losses)
- [6. Restore a new image](#6-restore-a-new-image)
- [7. Resume](#7-resume)

</details>

## 1. Prepare pairs

```text
data/restoration/
├── hq/a.png
├── hq/b.png
├── lq/a.png
├── lq/b.png
└── test.png
```

HQ is the desired target; LQ is the actual input. Pair by stem and verify dimensions, composition and alignment. Hold test.png out for inference checks.

```bash
cp docs/examples/tutorial-restoration.toml restoration.toml
```

Replace model weights, VAE, text encoder and tokenizer sources as described in [model paths](getting-started.md#2-fill-model-paths).

## 2. Fill data and training settings

```toml
[dataset.groups.main]
kind = "restoration"
source = "data/restoration/hq"
low_quality_source = "data/restoration/lq"
cache_directory = "cache/restoration"
resolution = [1024, 1024]
batch_size = 1
prompt = ""
horizontal_flip_probability = 0.0

[train]
parameter_mode = "adapter"
learning_rate = 5e-5
max_updates = 100
accumulation_steps = 4
seed = 42

[train.adapter]
algorithm = "lora"
rank = 32
alpha = 32.0

[method]
kind = "restoration"

[method.restoration]
midtimestep = 400
calibrate = false
encoder_rank = 4

[losses]
lrr = 3.0
l1 = 0.5
dv3d = 0.0
gan = 0.0
fdl = 0.0
histogram = 0.0

[execution]
precision = "bf16"
recompute = "block"
```

This is a fixed-time CUDA BF16 starting recipe. 400 is an experimental comparison point, not a universally optimal time. prompt="" supplies empty text for all images; change the string for a shared caption without txt files. Do not add degradation tables for existing pairs.

encoder_rank=4 trains a separate VAE encoder adapter; 0 disables it. It exports independently of the denoiser LoRA and must be paired with that LoRA at inference.

## 3. Choose fixed time or calibration

| Policy | Fields | Use |
| --- | --- | --- |
| Fixed integer time | midtimestep=400, calibrate=false, no sigma | Controlled time-point experiments |
| Fixed sigma | sigma strictly within (0,1), calibrate=false | Reuse an estimated noise level |
| Automatic calibration | calibrate=true, optional calibration_pairs=32, no sigma | Estimate noise level from current pairs |

The model's schedule maps integer time to sigma; dividing by 1000 is not correct for every family. Automatic calibration selects an estimate instead of using the integer setting as a second training time.

To inspect calibration without training:

```bash
uv run --no-dev unituner mid-timestep restoration.toml --device cuda
```

This reads only the VAE and pairs, needs no cache, and does not load denoiser/text weights. It prints JSON without editing TOML. To retain its result, set sigma manually under method.restoration and disable calibrate. Reconsider calibration after changing data, dimensions or VAE.

## 4. Preview, cache and train

```toml
[output]
directory = "runs/restoration"
name = "restoration"
save_every_updates = 50
tensorboard = "runs/tensorboard"

[preview]
every_updates = 25
at_start = true
num_images = 1
```

Do not add preview.samples for restoration. Previews rotate through actual data and display LQ/restored/HQ comparisons. A fixed seed keeps selection order stable. Inspect input consistency, skin tones, texture and excessive sharpening.

```bash
uv run --no-dev unituner validate restoration.toml
uv run --no-dev unituner cache restoration.toml --device cuda
uv run --no-dev unituner train restoration.toml --device cuda
```

Restoration reads original pairs and runs the trainable encoder/image losses live. Keep source images after caching.

Main outputs after 100 updates:

```text
runs/restoration/
├── weights/
│   ├── restoration-00000050.safetensors
│   ├── restoration-00000050.encoder.safetensors
│   ├── restoration-00000100.safetensors
│   ├── restoration-00000100.encoder.safetensors
│   ├── restoration.safetensors
│   └── restoration.encoder.safetensors
├── resume/restoration.state-00000100.safetensors
├── previews/
└── logs/restoration.jsonl
```

Defaults keep every inference export and only the newest resume state. See [saving](training/README.md#saving-and-resuming) for retention limits.

## 5. Add perceptual, color or adversarial losses

After validating the basic workflow, enable only objectives you need:

| Field | Effect | Additional setting |
| --- | --- | --- |
| losses.lrr | Latent restoration target | No external perceptual weights |
| losses.l1 | Pixel agreement | None |
| losses.dv3d | DINOv3 feature distance | method.restoration.dino_weights |
| losses.gan | Adversarial details and discriminator updates | The same DINOv3 ConvNeXt weights |
| losses.fdl | VGG frequency-distribution target | method.restoration.vgg_weights |
| losses.histogram | Color distribution | Optional histogram_bins; no feature checkpoint |

DINO here means ConvNeXt, not an arbitrary DINO ViT. Default widths/depths match ConvNeXt Large; VGG channel defaults match VGG19. Match actual topology, not filenames alone.

See [restoration.toml](../examples/restoration.toml) for a perceptual recipe and [oil-skin-lora.toml](../examples/oil-skin-lora.toml) for Z-Image Turbo skin restoration. Replace its environment-specific paths.

TensorBoard records generator, discriminator and component losses at every update. Lower total loss does not guarantee every component or visual quality improves. Changing coefficients changes total-loss scale; compare fixed previews and earlier exported versions when artifacts appear.

## 6. Restore a new image

```bash
cp docs/examples/tutorial-restore.toml restore.toml
uv run --no-dev unituner restore restore.toml --device cuda
```

Edit root input/output/prompt and model paths. adapter.weights and encoder_weights must reference the same update. Numbered files prevent version drift while training continues exporting.

```toml
input = "data/restoration/test.png"
output = "runs/restoration/result.png"
prompt = ""
encoder_weights = "runs/restoration/weights/restoration-00000100.encoder.safetensors"
```

Artifacts store sigma, so inference normally needs no override. Restoration uses posterior means and preserves input dimensions. Remove encoder_weights if encoder_rank=0; for full training use the exported base as model.weights and remove adapter tables.

## 7. Resume

```bash
uv run --no-dev unituner train restoration.toml --device cuda --resume runs/restoration/resume/restoration.state-00000050.safetensors
```

This applies to an interrupted run where update 50 remains the newest state. After finishing update 100, only that final state remains by default. Resume includes effective sigma, encoder, discriminator, optimizers, RNG and data cursor; it does not recalibrate. An inference LoRA cannot replace a resume file.

---

[Documentation](README.md) · [Parameter reference](training/parameters.md) · [Troubleshooting](troubleshooting.md)
