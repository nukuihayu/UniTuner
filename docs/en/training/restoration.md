# Image restoration training

[简体中文](../../training/restoration.md) · **English**

[Documentation](../README.md) · [Training](README.md) · [Examples](../examples/README.md)

Mid-timestep restoration feeds the encoded low-quality (LQ) latent directly into the model at a fixed intermediate time, then performs one restoration step. It adds no extra noise to the LQ latent. Models retain their native timestep units and velocity direction. Training, previews, checkpoints, and `unitunner restore` share the same formula across eight image families, with adapter or full parameter training.

<details>
<summary>On this page</summary>

- [Parameters and loss terms](#parameters-and-loss-terms)
- [Local periodic residual constraint](#local-periodic-residual-constraint)
- [Models and posterior policy](#models-and-posterior-policy)
- [Data and commands](#data-and-commands)
- [Mid-timestep settings](#mid-timestep-settings)
- [Relationship to OMGSR](#relationship-to-omgsr)
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
| `dino_weights`, `vgg_weights` | Unset | Local feature-network weights required by enabled losses |
| `dino_widths` | `[192, 384, 768, 1536]` | DINOv3 ConvNeXt-Large by default |
| `dino_depths` | `[3, 3, 27, 3]` | DINO four-stage depths |

Both `dino_weights` and `vgg_weights` accept `.pt`, `.pth`, and `.safetensors` without conversion or renaming. PyTorch files are read with `weights_only=True` and must contain flat tensor state dicts. DINO defaults match the official `dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth`; other sizes require matching explicit `dino_widths` and `dino_depths`. VGG19 accepts the official `vgg19-dcbb9e9d.pth` (excluding its known classifier) or feature-only weights. Unknown keys and missing feature weights still fail.

| `[losses]` field | Default weight | Supervision | Extra asset |
| --- | --- | --- | --- |
| `lrr` | `5.0` | MSE between LQ latents and noisy frozen HQ latents | None |
| `l1` | `0.5` | Pixel absolute error between restored and HQ images | None |
| `dv3d` | `3.0` | DINOv3-ConvNeXt feature distance | DINO weights |
| `gan` | `0.5` | Adversarial image supervision with discriminator updates | DINO weights |
| `fdl` | `0.5` | Feature frequency distribution | VGG19 weights |
| `histogram` | `0.0` | Normalized Lab color CDF matching | None |
| `ms_ssim` | `0.0` | Five-scale RGB structural similarity: `1 - MS-SSIM` | None |
| `phase` | `0.0` | Target-texture-protected periodic residual constraint; disabled by default | None |

These are schema defaults, not a universal recommended recipe. Explicitly zero unused terms: omitting DINO/VGG paths while leaving their default losses enabled is invalid. At least one image-output term (`l1`, `dv3d`, `gan`, `fdl`, `histogram`, or `ms_ssim`) must be nonzero, or `[method.wavelet]` must enable an effective band. Neither LRR nor phase alone can supervise the complete decoded result.

For large datasets, startup indexing validates source images, captions, and cache files one record at a time, reading only cache header metadata rather than latent/text tensors. Dataset fingerprints stream individual records instead of retaining extra identity dictionaries and a complete JSON copy; the necessary file-pair index still scales with image count. Cache formats and resume fingerprints remain unchanged, so caches need not be rebuilt. The terminal displays per-image validation progress.

Linux system/container RAM counters may also include file page cache. After fully hashing each HQ, LQ, and cache file, restoration indexing automatically requests release of its scanned pages with `POSIX_FADV_DONTNEED`, reducing cache buildup from the one-time scan. No configuration or administrator privileges are required. This is a best-effort kernel/filesystem hint; mapped, active, or dirty pages may remain. Unsupported platforms continue validation normally. Global caches are never flushed and content validation is never skipped. Later training reads may need to fetch these files from disk again.

Restoration training also applies this policy when hashing model, VAE, and feature artifacts, including sharded weights, without changing digests or resume compatibility. DINO/VGG ZIP-format PyTorch checkpoints use private source-file memory mappings when loading onto an accelerator, avoiding a complete anonymous CPU checkpoint copy. Legacy pickle files retain ordinary loading. Touched mapped pages may still count toward container RAM.

All eight image families (Z-Image, FLUX.1, FLUX.2, FLUX3, MOSS, Qwen Image, Qwen Image2.1, and Krea2) share these optimizations: training/previews read only cached text conditions, plus FLUX.1 pooled features, preserving Krea2 selected-layer text without transferring unused image/reference latents; RGB decoding avoids extra full-image PIL copies, pads only when necessary, and creates a single HQ crop for synthetic degradation; batch size one avoids an extra image stack, and CPU crops are released after device transfer; generator latent/graph references are cleared before the discriminator phase once backward has completed. No configuration changes are needed. Crop/degradation RNG, loss weights, and model equations are preserved. Model, optimizer, and necessary activation storage are still required.

## MS-SSIM

Enable with `[losses] ms_ssim = 0.1` (an example weight, not a tuned recommendation). The default is `0.0`, which skips computation and preserves existing resume identities. Enabled weights and the numerical policy are included in resume identity. The detached `ms_ssim` metric records the unweighted loss.

The loss is `1 - mean(MS-SSIM)` over samples and RGB channels, using five scales with exponents `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`, an 11×11 Gaussian window with sigma 1.5, valid filtering, and K1=0.01/K2=0.03. Between scales it uses 2× average pooling with parity padding for odd dimensions. Both image dimensions must exceed 160 pixels; configuration validation rejects smaller training crops. It does not silently reduce the scale count.

Statistics and reductions run in FP32 even during BF16/FP16 training. RGB is mapped from nominal `[-1,1]` to `[0,1]` without clipping decoder overshoots. Each scale score is clamped to `[1e-6,1]` before fractional powers, avoiding non-finite gradients for negative similarity; gradients saturate below that floor. HQ is detached. No extra feature networks or dependencies are required.

MS-SSIM supervises structure across scales; it is not a dedicated periodic-artifact penalty. Combining it with L1 can retain pixel/color supervision. It is accepted as a standalone image objective, but numerical verification does not establish better restoration quality or guarantee that grids disappear. Tune its weight using held-out images.

## Wavelet MSE

Wavelet MSE uses `[method.wavelet]`: decompose output and HQ recursively, compute element-mean MSE per enabled band and level, then sum band coefficients and multiply by `weight`. It is not a single MSE averaged across all coefficients. `normalize_bands = false` preserves absolute band amplitudes. Restoration applies no timestep fade; `timestep_cutoff` and `timestep_transition_width` apply only to flow matching. Wavelet loss requires no external feature weights.

```toml
[losses]
lrr = 0.0
l1 = 0.0
dv3d = 0.0
gan = 0.0
fdl = 0.0
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

`losses.phase` is a supplementary loss, disabled by default, for repeated phase bias relative to HQ. The implementation protects both local lighting planes and target-texture contrast instead of treating all such errors as grid artifacts.

Within each window, fit and remove a plane from both `e = generated - target` and the target, obtaining `e_t` and `y_t`. Then compute:

```text
beta = mean(e_t * y_t) / (mean(y_t²) + 1e-6)
r = e_t - beta * y_t
m = means of r grouped by coordinates modulo phase_period, then centered
phase = mean(sqrt(m² + phase_epsilon²) - phase_epsilon)
```

`beta` is a regularized local contrast fit. The variance ridge `1e-6`, in `[-1,1]` RGB units, keeps flat targets well conditioned. Backpropagation includes the fitted coefficient; only targets are detached. Statistics use FP32, with no extra feature-network forward or inference changes. This target-dependent linear transform followed by a robust penalty is not a single orthogonal projector.

Correctly reconstructed periodic target textures incur no penalty. Local errors proportional to target texture are penalized less, leaving their supervision to L1/perceptual objectives. Misaligned edges and details can still be affected. Brightness protection in output space also does not guarantee unchanged brightness after training: shared model parameters couple error directions.

Merge these fields into an existing recipe, keeping its other loss and model settings:

```toml
[losses]
phase = 4.0

[method.restoration]
phase_period = 16
phase_cycles = 8
phase_epsilon = 0.001
```

This uses `16 × 8 = 128` pixel windows; both training dimensions must be at least 128. Interior windows do not overlap. Incomplete edges get full windows touching those edges and overlapping their neighbors. No padding or discarded pixels; every window has equal weight. Period 16 is a Z-Image candidate, not an automatic model-specific choice. Select it from measured output periodicity instead of blindly stacking 2/4/8/16. This loss cannot supervise the complete image alone or repair misaligned training pairs.

Real-photo comparisons used Z-Image Turbo, official DINOv3 ConvNeXt Large, 512 crops, 80 training photos and 20 independent validation photos (40 crops), with native synthetic degradations. Each run used 300 updates, accumulation 4, learning rate `5e-5`, LoRA rank/alpha 8/8, encoder rank 4, explicit sigma 0.121, BF16, empty text conditioning, and no EMA. LRR/Dv3D/L1/GAN weights were 5/3/0.5/0.5; FDL and histogram were zero. Initialization, sample order, and noise were paired within each seed.

| Seed | Local phase RMS change | PSNR (baseline → enabled) | Dv3D (baseline → enabled; lower is better) | SSIM (baseline → enabled) |
| --- | --- | --- | --- | --- |
| 123 | −26.1% | 26.495 → 26.734 dB | 0.12546 → 0.12450 | 0.6615 → 0.6810 |
| 456 | −9.2% | 27.264 → 27.288 dB | 0.12235 → 0.12079 | 0.69944 → 0.69951 |

All 20 validation photos improved in local phase error in both runs; mean smooth-region error fell by 26.4% and 11.2%. However, not every image or metric improved: edge error increased by about 1.3% in the second seed, whose mean SSIM was essentially unchanged; one source photo lost about 1 dB in mean PSNR. The earlier plane-only constraint lost about 0.39 dB in that seed, so only the target-texture-protected implementation is retained. Raising L1 from 0.5 to 1.0 also failed to consistently outperform each baseline; default L1 is unchanged.

These are exploratory early-training comparisons with two seeds and synthetically degraded real photos. Validation data informed method selection; this is not an independent final test set or an official DIV2K benchmark. The results do not guarantee removal of all user artifacts or universally better converged quality. The loss remains disabled by default; 4.0 is only a tested starting point. Monitor repeated patterns, real details, edges, and colors together. Disabled phase preserves earlier resume identities; enabling or changing an effective loss parameter changes the identity and cannot silently resume the previous objective's full training state.

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

With LRR enabled, the shared step runs frozen HQ posterior encoding, LRR Gaussian noise, then trainable LQ posterior encoding. Sampling VAEs draw noise in logical shape so channels-last layout does not change the random-value/element mapping. Mean VAEs do not add posterior sampling. This follows the reference Trainer's effective LRR path while retaining native normalization, time, and prediction signs.

Posterior sampling adds randomness without guaranteeing quality or speed improvements. Explicit mean supports deterministic variants. Disabling LRR skips unused HQ encoding and random draws, so stepwise RNG parity with the reference Trainer is not promised. Posterior/random-order changes cannot be silently resumed as the same trajectory. Previews and restoration inference use the posterior mean.

FLUX Kontext/Fill, Qwen edit/layered, Wan, and H3 are outside restoration support. FLUX.2/FLUX3/MOSS use native packed-BN mean encoding and do not expose unverified posterior sampling.

## Data and commands

Start with [FLUX](../../examples/flux-restoration.toml), [FLUX.2](../../examples/flux2-restoration.toml), [FLUX3](../../examples/flux3-restoration.toml), or [MOSS](../../examples/moss-image-restoration.toml). These use LRR + L1 without perceptual network weights.

`dataset.groups.main.source` holds HQ images; `dataset.groups.main.low_quality_source` holds aligned LQ images matched by stem. Synthetic degradation and mixed sources follow the [training guide](README.md#restoration-and-intermediate-time). Caching extracts native text features; FLUX keeps T5 and pooled CLIP features. CLIP resides in the bundle's `text_encoder/`; `model.assets` identifies that bundle root, defaulting to the denoiser's parent.

```bash
uv run unitunner cache docs/examples/flux3-restoration.toml --device cuda
uv run unitunner train docs/examples/flux3-restoration.toml --device cuda
uv run unitunner train docs/examples/flux3-restoration.toml --device cuda --resume runs/flux3-restoration/resume/model.state-00000050.safetensors
```

For full parameters, set `train.parameter_mode = "full"` and remove `[train.adapter]`. As in ordinary flow matching, set `[execution] precision = "bf16"` for BF16 AMP in the VAE, DINO, VGG and discriminator convolutions and linear layers. Trainable master parameters and optimizer states follow the shared flow-matching precision policy; FFT and loss statistics retain FP32. Full mode requires floating base weights; adapters can use a [quantized frozen base](quantized-base.md). Full mode uses resident weights and supports block recomputation/CUDA compilation. Frozen-block H2D supports adapters only. LayerSync, REPA, and TREAD are rejected in restoration.

## Mid-timestep settings

```toml
[method]
kind = "restoration"

[method.restoration]
midtimestep = 244
```

```bash
uv run unitunner train train.toml --device cuda --midtimestep 244
```

The integer range is 1..998 and applies only to restoration. An explicit CLI value overrides TOML; 244 is used only when both omit the value. Optional `method.restoration.sigma` directly specifies the noise fraction; `calibrate=true` estimates it from paired data.

Selection order is calibration, explicit sigma, then integer mid-timestep. Calibration and explicit sigma are mutually exclusive. The CLI changes only the integer and does not cancel another time policy. No special restrictions arise from choosing an integer other than 244.

FLUX.1 uses a reverse index into the native 999-step shifted schedule:

```text
build_schedule(999, (W//16)*(H//16))[-(midtimestep+1)]
```

W/H are configured crop dimensions in pixels. Therefore FLUX.1 integer 244 **does not mean sigma 0.244**. Other current families use `midtimestep/1000`, retaining their native units and velocity direction. This is a starting convention rather than an independently optimized time for every model.

Time conversion preserves native operation order, including FLUX's `full(sigma*1000)/1000`. FLUX.1 restoration uses guidance embedding 1.0, matching Trainer OMGSR-F without changing ordinary generation or flow matching. FLUX3 distilled keeps identical target/source training times from its original fine-tuning entry point.

## Relationship to OMGSR

Based on [OMGSR v2 §3 and supplement §6](https://arxiv.org/html/2508.08227v2):

- Feed LQ latents directly, without added noise; train encoder adapters to narrow the representation gap.
- The FM LRR target is `(1-sigma)*encode(HQ) + sigma*noise`, with frozen HQ encoding and MSE supervision.
- Noise-minus-data models restore as `low - sigma*velocity`; Z-Image's sign is handled at the model boundary.
- Calibration minimizes the mean absolute per-image SNR difference in equations 18/19. FLUX.1 candidates use the same shifted grid.
- Frozen DINOv3-ConvNeXt's first three stages provide Dv3CD and multilevel discrimination; the generated-image branch retains gradients.

`[execution] recompute = "block"` also checkpoints DINO downsampling layers and ConvNeXt blocks, reducing activation memory for Dv3CD and generator GAN losses at the cost of extra backward recomputation. HQ features and the frozen backbone during discriminator updates need no image gradients and skip recomputation. Discriminator heads are not checkpointed, avoiding repeated spectral-normalization state updates. Set `recompute = "none"` to disable this memory/time tradeoff. DINO channel normalization uses native LayerNorm with FP32 accumulation for BF16 statistics; small numerical differences from elementwise BF16 normalization are expected.

With `recompute = "block"`, restoration also checkpoints stateless VAE residual, spatial-attention and applicable resampling blocks to reduce high-resolution activation memory. The VAE stays in eval mode; decoder weights remain frozen and encoder LoRA gradients are preserved. Residual checkpointing covers Z-Image, the FLUX family and Qwen Image 2.1. Qwen Image/Krea2 temporal-history blocks keep their original path; only stateless spatial attention is checkpointed. Recomputation increases compute time and does not eliminate convolution workspace or DiT attention memory requirements.

Training Inductor compilation scopes `triton.cudagraphs=false` and `epilogue_fusion=true` per graph, avoiding CUDA Graph private pools while retaining pointwise fusion without mutating process-global settings. The current compilation entry point targets DiT, not VAE/DINO. Maximum GEMM autotuning, coordinate-descent tuning and 1x1-convolution-to-matmul conversion are not enabled by default: benefits depend on the graph and hardware and can increase initial compilation time.

The paper's 244 comes from FLUX.1-dev at 1K; its SD2.1-base result is 273. The reference Trainer's Z-Image default 121 is not a paper conclusion for Z-Image. UniTunner uses 244 as a common initial configuration value. Recalibrate when changing model, resolution, VAE, data, or degradation distribution.

DDPM equation 7 uses alpha-bar reconstruction and cannot be replaced with the FM Euler formula. These entry points do not introduce DDPM support.

The project recipe keeps LRR/Dv3D/L1/GAN/FDL weights of 5/3/0.5/0.5/0.5, with histogram loss defaulting to zero. FDL, histogram, and phase are recipe extensions rather than original OMGSR losses or evidence of reproduced paper quality.

FDL uses orthonormal FFTs (`norm="ortho"`) to normalize amplitude scaling for each feature-map size. Phase, all 256 projections, and the sum over levels are preserved. This is a variant of the original unnormalized FDL and is not bounded to [0,1]. Old FDL values are not directly comparable; reassess `losses.fdl` when continuing an older run.

Calibration uses the unadapted VAE with the configured posterior. FLUX defaults to sample, matching the official implementation. Isolated seeded posterior RNG leaves training RNG unchanged. Explicit mean is a deterministic variant. Candidate grids exclude endpoints 0 and 1, where SNR is undefined. SNR errors are averaged per image rather than calculated over a mixed batch. Effective sigma is saved in checkpoints and publications; resume does not recalibrate and inference reads the saved value.

## Outputs and one-step inference

Outputs include the denoiser model/adapter, an optional encoder adapter, and resumable training state. Model/adapter and encoder publications record sigma/posterior; inference checks sigma consistency. For full inference, point `model.weights` to the trained model. For LoRA, retain the base model and configure `[adapter]` / `[adapter.config]`. `encoder_weights` points to the trained encoder delta. One-step inference uses posterior means and preserves input dimensions.

## Standalone calibration

```bash
uv run unitunner mid-timestep docs/examples/restoration-calibration.toml --device cuda
```

See the [calibration example](../../examples/restoration-calibration.toml) and [full restoration example](../../examples/restoration-full.toml). The command loads images and the local VAE, without the denoiser, text encoder, perceptual networks, optimizer, or precomputed cache. It follows configured pair alignment, synthetic degradation, crop dimensions, and `train.seed`.

By default it uses the first 32 unique pairs. Configure `calibration_pairs = 128` under `[method.restoration]`, with `calibrate=true`, to expand the sample. JSON output includes `midtimestep`, `sigma`, `mean_snr_error`, `pairs`, `posterior`, and `sampling`. The integer is an index; sigma is the actual noise fraction.

Either write sigma into `method.restoration.sigma` and disable calibration, or enable calibration to estimate once before training. For non-FLUX results, copying sigma preserves FP32 candidate-grid rounding that the integer alone does not encode.

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

Standalone and pre-training calibration share `estimate_mid_timestep`, accumulating equations 18/19 statistics without retaining all latents. Each pair needs two VAE encodings and a vectorized candidate-error evaluation; the network is not rerun per timestep. All eight current restoration families use their native normalized latent space. New architectures with different schedules need an explicit candidate mapping.

The restoration objective uses the [OMGSR method](https://github.com/wuer5/OMGSR). Project loss defaults are listed in [parameters](parameters.md#losses); configure coefficients explicitly when using a different recipe.

FDL retains all 256 projections and full spatial sorting, processing eight projection channels at a time. Backward recomputes projections and sorting instead of retaining all sort indices. The generated-image VGG branch also recomputes activations, trading compute for memory without extra configuration.


---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
