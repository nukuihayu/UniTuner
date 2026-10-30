# Feature objectives and token routing

[简体中文](../../training/objectives.md) · **English**

[Documentation](../README.md) · [Training](README.md) · [Examples](../examples/README.md)

LayerSync, REPA, and TREAD can be combined for flow matching across ten families, in adapter or full mode. They are not connected to restoration. Layer windows must match the actual native topology. [Execution strategies](execution.md) independently control compilation, recomputation, and residency.

<details>
<summary>On this page</summary>

- [Choose components](#choose-components)
- [What each addition changes](#what-each-addition-changes)
- [Configuration](#configuration)
- [Layers and routing](#layers-and-routing)
- [Features and gradients](#features-and-gradients)
- [REPA caches](#repa-caches)
- [Wavelet loss](#wavelet-loss)

</details>

## Choose components

| Method | Purpose | Extra resources |
| --- | --- | --- |
| LayerSync | Negative cosine alignment of early student features and a later stop-gradient teacher | No external teacher weights |
| REPA | Trainable projection of native target features into frozen encoder features | Matching DINOv3 ViT weights, digest, and feature cache |
| TREAD | Route a subset of target tokens through a layer window | Per-forward routing RNG; no external encoder |

## What each addition changes

LayerSync and REPA add supervision to the base flow loss:

```text
L_total = L_flow + w_sync · L_sync + w_repa · L_repa
```

LayerSync uses a later layer of the same model as its teacher, so it needs no external network. REPA supplies a separate pretrained visual representation and therefore requires an encoder identity and matched target-feature cache. TREAD changes which target tokens traverse a layer window; it does not add another loss term. Routing half the target tokens does not imply halving the full training workload, since text, references, and other work remain.

Begin with a working flow baseline, enable one method, and compare held-out outputs, memory, and throughput. Only combine methods after checking their individual behavior. Remove an optional method's section to disable it; positive-weight validation means `weight=0` is not a disable switch.

| Parameter | Default or requirement | Meaning |
| --- | --- | --- |
| LayerSync `weight` | `1.0`, positive | Auxiliary loss coefficient |
| `student_layer`, `teacher_layer` | `0`, `1` | Zero-based indices; teacher follows student |
| REPA `weight`, `layer` | `1.0`, `0` | Projection loss coefficient and capture layer |
| `encoder_dim` | Required positive integer | Must match actual encoder feature width |
| `encoder_sha256` | Required 64 lowercase hex characters | Pins the encoder's content |
| TREAD `drop_fraction` | `0.5`, strictly between 0 and 1 | Fraction of target tokens bypassing the selected window |
| `start_layer`, `end_layer` | `0`, `1` | Nonempty half-open layer window |

## Configuration

Add the required sections to a training TOML; remove unused methods. See [flow-features.toml](../../examples/flow-features.toml) for the full template. Replace the placeholder digest before actual caching.

```toml
[method.tread]
drop_fraction = 0.5
start_layer = 0
end_layer = 2

[method.layersync]
weight = 0.2
student_layer = 0
teacher_layer = 1

[method.repa]
weight = 0.3
layer = 0
encoder_dim = 768
encoder_weights = "/absolute/path/dinov3-vit.safetensors"
# Replace with the actual lowercase SHA256 of encoder_weights.
encoder_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
encoder_device = "cpu"
cache_directory = "/absolute/path/repa-cache"
```

## Layers and routing

Layer indices start at zero and count main visual/joint blocks. TREAD's end layer is exclusive. Windows cannot exceed the native block count; LayerSync's teacher must follow its student.

TREAD samples an ordered target subset once per forward. Unselected tokens use a differentiable residual bypass. Text, separate references, audio, and padding retain their original order. Previews disable routing and do not consume routing RNG. All native supervision slots in Wan independent-frame mode participate in flow matching, with REPA features in the same coordinate order; there is no RCM mask.

## Features and gradients

LayerSync stops teacher gradients and uses negative cosine loss. REPA uses training-only FP32 LayerNorm/Linear projectors and stop-gradient targets, with trilinear spatiotemporal-grid interpolation. Feature capture is scoped to the current forward and removed before backward; recomputation does not capture twice. Native model weight keys remain intact. All three methods can be combined; restoration rejects them.

## REPA caches

During `unituner cache train.toml --device cuda`, REPA receives exactly the VAE's target crop and frame order. It does not decode latents or apply an independent center crop. Native or official MLP DINOv3 ViT-S/B/L safetensors are supported with bicubic/antialias preprocessing, ImageNet normalization, and per-frame encoding. CLS/register tokens are excluded. Native alpha remains in caches/supervision; DINO receives RGB.

Encoder content must match its configured digest and feature width. Training uses the feature cache without loading DINO. Callers providing precomputed features through `save_repa_target` and `save_repa_manifest` must guarantee correct crop/frame correspondence.

The manifest binds the full digest of every native cache record. Training checks coverage and content; resume identity includes the manifest digest. Projectors are restored with checkpoints and do not enter inference adapter publications.

## Wavelet loss

Eight image families can add DWT band loss to flow matching or restoration, in adapter or full mode. Flow matching supervises native velocity prediction/target; restoration supervises decoded generated/HQ images. Wan and H3 do not use this image objective.

```toml
[method.wavelet]
weight = 0.5
wavelet = "db4"
levels = 3
ll = 0.1
lh = 0.01
hl = 0.01
hh = 0.05
ll_level_threshold = -1
normalize_bands = false
timestep_cutoff = 0.7
timestep_transition_width = 0.4
```

Omitting the section adds no computation. The implementation derives from the DWT branch of [rockerBOO/wavelet-loss](https://github.com/rockerBOO/wavelet-loss), with built-in Haar/db4 filters and no extra loss library.

It uses zero boundaries, full convolution followed by `1::2` subsampling, recursive LL decomposition, and weighted element-mean MSE per band. LH means high-pass H / low-pass W; HL means low-pass H / high-pass W. Negative LL thresholds resolve as `levels + threshold`; the default includes LL only at the last level.

Optional normalization shares the target mean/std between prediction and target, preserving prediction amplitude errors. Single-coefficient bands use zero variance to avoid undefined unbiased-std NaNs.

Flow matching weights each sample by `sigmoid((cutoff-sigma)*4/transition_width)`, using the `[0,1]` noise fraction rather than native timestep units. Layered images are transformed individually and averaged. Restoration has no noise-time decay. All band operations run FP32 outside autocast, with gradients available to both prediction and target in standalone/flow use. Restoration detaches the HQ target, matching its other supervised image losses.

SWT, experimental QWT, SNR-aware Huber, and per-band/per-level override weights are not exposed. Extra objectives add computation without guaranteeing convergence or quality improvements.

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
