# Image restoration training

[简体中文](../../training/restoration.md) · **English**

[Documentation](../README.md) · [Training](README.md) · [Examples](../examples/README.md)

Mid-timestep restoration feeds the encoded low-quality (LQ) latent directly into the model at a fixed intermediate time, then performs one restoration step. It adds no extra noise to the LQ latent. Models retain their native timestep units and velocity direction. Supported image models offer adapter or full-parameter training. Use the trained weights with `unituner restore`.

<details>
<summary>On this page</summary>

- [Parameters and loss terms](#parameters-and-loss-terms)
- [Local periodic residual constraint](#local-periodic-residual-constraint)
- [Models and posterior policy](#models-and-posterior-policy)
- [Data and commands](#data-and-commands)
- [Mid-timestep settings](#mid-timestep-settings)
- [High-resolution restoration](#high-resolution-restoration)
- [Outputs and one-step inference](#outputs-and-one-step-inference)
- [Standalone calibration](#standalone-calibration)

</details>

## Parameters and loss terms

For a complete paired-data run, begin with the [restoration tutorial](../restoration-tutorial.md). This page explains the time/posterior policy and the reference formulas.

| `method.restoration` field | Default | Role |
| --- | --- | --- |
| `midtimestep` | `244` | Integer time index, from 1 through 998 |
| `sigma` | Unset | Explicit noise fraction strictly between 0 and 1 |
| `calibrate` | `false` | Estimate sigma before training; mutually exclusive with explicit sigma |
| `calibration_pairs` | `32` | Maximum encoded crop/tile pairs |
| `calibration_sampling` | `"random"` | One random crop per image, or `tiles` at dataset resolution |
| `encoder_rank` | `4` | Encoder adapter rank; 0 disables adaptation |
| `posterior` | Native policy | `sample` or `mean` where the VAE supports it |
| `histogram_bins` | `32` | Color histogram bins, at least 2 |
| `phase_period`, `phase_cycles` | `16`, `8` | Period and complete cycles per window axis; both at least 2 |
| `phase_epsilon` | `0.001` | Charbonnier smoothing scale in `[-1,1]` RGB units; `1e-12` to `1.0` |
| `dino_weights` | Unset | Local feature-network weights required by enabled losses |
| `dino_widths` | `[192, 384, 768, 1536]` | DINOv3 ConvNeXt-Large by default |
| `dino_depths` | `[3, 3, 27, 3]` | DINO four-stage depths |

`dino_weights` accepts `.pt`, `.pth`, and `.safetensors` without conversion or renaming. PyTorch files are read with `weights_only=True` and must contain flat tensor state dicts. DINO defaults match the official `dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth`; other sizes require matching explicit `dino_widths` and `dino_depths`.

| `[losses]` field | Default weight | Supervision | Extra asset |
| --- | --- | --- | --- |
| `lrr` | `5.0` | MSE between LQ latents and noisy frozen HQ latents | None |
| `l1` | `0.5` | Pixel absolute error between restored and HQ images | None |
| `dv3d` | `3.0` | DINOv3-ConvNeXt feature distance | DINO weights |
| `gan` | `0.5` | Adversarial image supervision with discriminator updates | DINO weights |
| `histogram` | `0.0` | Normalized Lab color CDF matching | None |
| `ms_ssim` | `0.0` | Five-scale RGB structural similarity: `1 - MS-SSIM` | None |
| `phase` | `0.0` | Target-texture-protected periodic residual constraint; disabled by default | None |

These are configuration defaults, not a universal recommended recipe. Explicitly zero unused terms: omitting DINO paths while leaving their default losses enabled is invalid. At least one image-output term (`l1`, `dv3d`, `gan`, `histogram`, or `ms_ssim`) must be nonzero, or `[method.wavelet]` must enable an effective band. Neither LRR nor phase alone can supervise the complete decoded result.

FDL has been removed. Delete `fdl` from `[losses]` and `vgg_weights`/`vgg_widths` from `[method.restoration]` in old configurations. These fields are no longer accepted, including `fdl = 0.0`.

Dataset checks can take time on large collections. Watch the terminal progress and allow enough host RAM for the image-pair index and model weights.

## MS-SSIM

Set `[losses] ms_ssim = 0.1` to add multiscale structural supervision; the default is `0.0`. The weight is an example, so tune it on held-out images. Both crop dimensions must exceed 160 pixels. No extra network weights are required.

The logged `ms_ssim` value is the unweighted loss; lower is better. Combine with L1 when pixel and color fidelity matter. Check real details as well as structure when choosing the weight.

## Wavelet MSE

Wavelet MSE uses `[method.wavelet]`: decompose output and HQ recursively, compute element-mean MSE per enabled band and level, then sum band coefficients and multiply by `weight`. It is not a single MSE averaged across all coefficients. `normalize_bands = false` preserves absolute band amplitudes. Restoration applies no timestep fade; `timestep_cutoff` and `timestep_transition_width` apply only to flow matching. Wavelet loss requires no external feature weights.

```toml
[losses]
lrr = 0.0
l1 = 0.0
dv3d = 0.0
gan = 0.0
histogram = 0.0
phase = 0.0

# Wavelet supervision settings.
[method.wavelet]
weight = 0.5
wavelet = "haar"
levels = 3
ll = 0.1
lh = 0.01
hl = 0.01
hh = 0.05
ll_level_threshold = -1
normalize_bands = false
```

This is a configuration example, not an optimized weight recommendation. For wavelet MSE alone, leave at least one effective band enabled. `ll_level_threshold = -1` includes LL only at the last level, while detail bands are supervised at every level. Wavelet metrics already include `weight`. See [all wavelet parameters](parameters.md#methodwavelet).

## Local periodic residual constraint

`losses.phase` is an optional auxiliary loss for periodic or grid-like artifacts relative to the HQ target. It is disabled by default and needs no extra feature weights. Add these fields to your existing recipe:

```toml
[losses]
phase = 4.0

[method.restoration]
phase_period = 16
phase_cycles = 8
phase_epsilon = 0.001
```

Both crop dimensions must be at least `phase_period * phase_cycles`, or 128 pixels in this example. Select the period from the artifacts visible in your output; it is not chosen automatically for each model. The weight shown is a starting example, not a general recommendation.

Keep a primary image loss such as L1 or a perceptual loss. Compare held-out images for repeated patterns, real texture, edges, and color; reduce or disable phase if detail suffers. It cannot correct misaligned training pairs. Changing effective loss settings may prevent resuming the previous full training state.

## Models and posterior policy

| Family | Supported variants | Training posterior |
| --- | --- | --- |
| Z-Image | Current native variants | `sample` by default; optional `mean` |
| FLUX | Dev | `sample` by default; optional `mean` |
| FLUX.2 | Native Dev / Klein 4B / Klein 9B base/distilled combinations | `mean` |
| FLUX3 | Base, distilled | `mean` |
| MOSS Image | Base, SFT | `mean` |
| Qwen Image | Original | `sample` by default; optional `mean` |
| Qwen Image 2.1 | Native image model | `sample` by default; optional `mean` |
| Krea2 | RAW, Turbo | `sample` by default; optional `mean` |

`posterior = "sample"` introduces VAE sampling randomness during training; `"mean"` uses the posterior mean. Only choose values supported by the model in the table. Previews and restoration inference use the mean. Recheck cache and resume compatibility after changing this setting.

FLUX Kontext/Fill, Qwen edit/layered, Wan, and MiniMaxH3 are outside this restoration method.

## Data and commands

Start with [FLUX](../../examples/flux-restoration.toml), [FLUX.2](../../examples/flux2-restoration.toml), [FLUX3](../../examples/flux3-restoration.toml), or [MOSS](../../examples/moss-image-restoration.toml). These use LRR + L1 without perceptual network weights.

`dataset.groups.main.source` holds HQ images; `dataset.groups.main.low_quality_source` holds aligned LQ images matched by stem. Synthetic degradation and mixed sources follow the [training guide](README.md#restoration-and-intermediate-time). Caching extracts native text features; FLUX keeps T5 and pooled CLIP features. CLIP resides in the bundle's `text_encoder/`; `model.assets` identifies that bundle root, defaulting to the denoiser's parent.

```bash
uv run unituner cache docs/examples/flux3-restoration.toml --device cuda
uv run unituner train docs/examples/flux3-restoration.toml --device cuda
uv run unituner train docs/examples/flux3-restoration.toml --device cuda --resume runs/flux3-restoration/resume/model.state-00000050.safetensors
```

For full parameters, set `train.parameter_mode = "full"` and remove `[train.adapter]`. As in ordinary flow matching, set `[execution] precision = "bf16"` for BF16 AMP in the VAE, DINO and discriminator convolutions and linear layers. Trainable master parameters and optimizer states follow the shared flow-matching precision policy; loss statistics retain FP32. Full mode requires floating base weights; adapters can use a [quantized frozen base](quantized-base.md). Full mode uses resident weights and supports block recomputation/CUDA compilation. Frozen-block H2D supports adapters only. LayerSync, REPA, and TREAD are rejected in restoration.

## Mid-timestep settings

```toml
[method]
kind = "restoration"

[method.restoration]
midtimestep = 244
```

```bash
uv run unituner train train.toml --device cuda --midtimestep 244
```

The integer range is 1..998 and applies only to restoration. An explicit CLI value overrides TOML; 244 is used only when both omit the value. Optional `method.restoration.sigma` directly specifies the noise fraction; `calibrate=true` estimates it from paired data.

Selection order is calibration, explicit sigma, then integer mid-timestep. Calibration and explicit sigma are mutually exclusive. The CLI changes only the integer and does not cancel another time policy.

FLUX.1 uses a reverse index into the native 999-step shifted schedule:

```text
build_schedule(999, (W//16)*(H//16))[-(midtimestep+1)]
```

W/H are configured crop dimensions in pixels. Therefore FLUX.1 integer 244 **does not mean sigma 0.244**. Other current families use `midtimestep/1000`, retaining their native units and velocity direction. This is a starting convention rather than an independently optimized time for every model.

## High-resolution restoration

If high-resolution training runs out of memory, reduce batch size or crop resolution, then enable recomputation:

```toml
[execution]
recompute = "block"
restoration_recompute = "block"
```

Recomputation saves activation memory at the cost of additional computation. Perceptual and adversarial losses also require memory; enable only the terms you need and provide their weights. See [execution settings](execution.md) and [precision settings](precision.md).

Recalibrate sigma after changing the model, VAE, resolution, dataset, or degradation settings. The calibrated sigma is saved in training outputs; resuming retains it rather than recalibrating.

## Outputs and one-step inference

Outputs include the denoiser model/adapter, an optional encoder adapter, and resumable training state. Model/adapter and encoder publications record sigma/posterior; inference checks sigma consistency. For full inference, point `model.weights` to the trained model. For LoRA, retain the base model and configure `[adapter]` / `[adapter.config]`. `encoder_weights` points to the trained encoder delta. One-step inference uses posterior means and preserves input dimensions.

## Standalone calibration

```bash
uv run unituner mid-timestep docs/examples/restoration-calibration.toml --device cuda
```

See the [calibration example](../../examples/restoration-calibration.toml) and [full restoration example](../../examples/restoration-full.toml). The command loads images and the local VAE, without the denoiser, text encoder, perceptual networks, optimizer, or precomputed cache. It follows configured pair alignment, synthetic degradation, crop dimensions, and `train.seed`.

By default it uses the first 32 unique pairs. Configure `calibration_pairs = 128` under `[method.restoration]`, with `calibrate=true`, to expand the sample. JSON output includes `midtimestep`, `sigma`, `mean_snr_error`, `pairs`, `posterior`, and `sampling`. The integer is an index; sigma is the actual noise fraction.

Either write sigma into `method.restoration.sigma` and disable calibration, or enable calibration to estimate once before training.

The standalone command accepts the training TOML directly, including the complete `[dataset.groups.<name>.degradation]` recipe. No pre-generated LQ files are needed for synthetic data. It extracts crops at `dataset.defaults.resolution` or the group override, applies the training degradation, then encodes each LQ/HQ pair for SNR statistics. Real pairs must already have matching aligned dimensions; they are not resized. Restoration uses fixed crops rather than text-to-image aspect-ratio buckets.

```toml
[method.restoration]
calibrate = true
calibration_pairs = 128
calibration_sampling = "tiles"

[dataset.defaults]
resolution = [512, 512]

[dataset.groups.main]
source = "/data/hq"
# Training still needs the other dataset fields; calibration needs no cache_directory.

[dataset.groups.main.degradation]
enabled = true
```

`tiles` traverses images in source order and tiles in row-major order. Small images are reflection-padded; the final row/column touches the edge and overlaps preceding tiles when necessary. Each tile receives its own configured flip and synthetic degradation before VAE encoding. `calibration_pairs` limits the **total tile pairs**, not source images. Tiles have equal weight, so larger images contribute more observations and later images are skipped when the limit is reached. Use the default `random` mode or increase the limit for broader source coverage. Fixed seeds reproduce sampling without advancing training RNG.

In a minimal calibration-only TOML, put `calibration_sampling`, `calibration_pairs`, and `posterior` at the top level; training TOMLs use `[method.restoration]`. All groups must still share a resolution to avoid mixing FLUX resolution-dependent time grids. These are independent calibration crops, not tiled VAE encoding with stitched latents, and they do not change normal training's random crop policy.

The restoration objective uses the [OMGSR method](https://github.com/wuer5/OMGSR). Project loss defaults are listed in [parameters](parameters.md#losses); configure coefficients explicitly when using a different recipe.


---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
