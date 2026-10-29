# Dataset configuration tutorial

[简体中文](../../training/datasets.md) · [Trainer](README.md) · [Parameters](parameters.md)

<details>
<summary>On this page</summary>

- [Keep data and training in one TOML](#keep-data-and-training-in-one-toml)
- [Multiple groups and defaults](#multiple-groups-and-defaults)
- [Caption sidecars and shared prompts](#caption-sidecars-and-shared-prompts)
- [Resolution and aspect-ratio buckets](#resolution-and-aspect-ratio-buckets)
- [Real LQ/HQ restoration pairs](#real-lqhq-restoration-pairs)
- [Editing, references, masks and layers](#editing-references-masks-and-layers)
- [Video datasets](#video-datasets)
- [When to rebuild caches](#when-to-rebuild-caches)

</details>

## Keep data and training in one TOML

A separate dataset file is unnecessary:

```toml
[model]
family = "z-image"
variant = "base"
weights = "/models/z-image.safetensors"
vae = "/models/ae.safetensors"
text_encoder = "/models/qwen3"

[dataset.groups.main]
kind = "image"
source = "/data/images"
cache_directory = "/data/cache/main"
resolution = [1024, 1024]
batch_size = 1

[train]
parameter_mode = "adapter"
learning_rate = 1e-4
max_updates = 1000

[train.adapter]
algorithm = "lora"
rank = 32
alpha = 32.0
```

This minimal configuration passes training schema validation. Caching also needs actual model, tokenizer and image files. The default method is flow matching; default training precision follows the base weights.

## Multiple groups and defaults

```toml
[dataset.defaults]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true
caption_extension = ".txt"

[dataset.groups.portraits]
kind = "image"
source = "/data/portraits"
cache_directory = "/data/cache/portraits"
repeats = 3

[dataset.groups.scenes]
kind = "image"
source = "/data/scenes"
cache_directory = "/data/cache/scenes"
resolution = [768, 512]
batch_size = 2
repeats = 1
```

defaults accepts shared sampling settings only: resolution, batch_size, repeats, enable_bucket, bucket_no_upscale, prompt and caption_extension. Paths, kind and conditions belong to each group. Group values override defaults.

Each portrait item appears three times per data pass and each scene once. Group frequency also depends on item counts and bucket batching. repeats is not an exact sampling percentage and does not change max_updates. Different group batch sizes can produce different sample counts per update.

Use ASCII identifier names such as main/portraits, without hyphens, spaces or leading digits. Cache directories must resolve to distinct locations, even when spelled differently.

## Caption sidecars and shared prompts

```text
/data/portraits/
├── a.png
├── a.txt
├── b.jpg
└── b.txt
```

| Caption policy | Configuration |
| --- | --- |
| Per-image text | Omit prompt; read same-stem caption_extension sidecars |
| Same caption for every item | Set prompt in defaults or the relevant group |
| Empty captions for every item | Explicit prompt="" overrides existing sidecars too |
| Shared text with group-specific overrides | defaults.prompt plus individual group.prompt values |

TOML has no null. Do not write prompt=null or use the string "null" to remove inheritance. When mixing shared-caption and sidecar groups, put prompt only in groups needing a shared caption.

Image directories are scanned directly, not recursively. Rebuild after editing images/captions. Shared prompts do not need duplicate txt files.

## Resolution and aspect-ratio buckets

resolution is [width,height] in pixels. enable_bucket selects crops by target area and input aspect ratio; disabling it uses one fixed-resolution crop. bucket_no_upscale avoids enlarging small images to larger buckets.

For multiscale training, set a list in defaults or an individual group:

```toml
[dataset.groups.main]
source = "./images"
cache_directory = "./cache/main"
resolution = [[1024, 1024], [768, 768], [512, 512]]
batch_size = 1
enable_bucket = true
prompt = "a portrait photo"
```

Every image produces a training sample at each scale; this does not randomly choose one scale per image. Each scale selects its own aspect buckets, or uses its exact dimensions with buckets disabled. Caches go to `cache/main/1024x1024`, `768x768` and `512x512`. Training mixes independent groups named `main__1024x1024`, etc. Each inherits repeats, batch_size, prompt and references. Mixing follows sample/batch counts rather than pixel area; different scales stay in separate batches. max_updates remains the total optimizer update count for the entire run.

A single `[1024,1024]` keeps using cache_directory directly. Nested lists, even with one scale, use scale subdirectories. Lists must contain unique positive integer pairs and cannot be empty. Rebuild caches after changing scales. Restoration training and mid-timestep calibration still require a shared single resolution; use multiscale lists for generation/editing training.

Most image families require 16-aligned dimensions; Qwen Image 2.1 and H3 require 32 alignment. Larger dimensions increase activation/attention cost. Reduce batch_size before comparing recomputation or lower resolution.

Condition images use the target bucket. Matching filenames do not register images with different compositions. Check LQ/HQ geometry before paired restoration.

## Real LQ/HQ restoration pairs

```toml
[dataset.groups.main]
kind = "restoration"
source = "/data/target"
low_quality_source = "/data/control"
cache_directory = "/data/cache/restoration"
resolution = [1024, 1024]
batch_size = 1
prompt = ""
horizontal_flip_probability = 0.0
```

source is HQ/target; low_quality_source is LQ/control. Match by filename stem, for example target/a.png to control/a.jpg. Horizontal flips apply to both images; set probability to 0 for direction-sensitive tasks.

Omit degradation when real LQ images already exist. To synthesize LQ, use dataset.groups.NAME.degradation with enabled=true and remove that group's low_quality_source. Separate groups may mix these policies; see [restoration-mixed](../../examples/restoration-mixed.toml). All restoration groups must share one resolution for calibration.

### Faster startup validation

Image-cache and restoration datasets use four workers by default for full validation, including image/cache hashes, caption matching and cache-header checks. Results retain input order and existing resume identities. At most twice the worker count is queued, without reading tensor payloads or queuing the whole dataset. Scanned file pages are still released with best-effort OS advice.

```toml
[train]
dataset_validation_workers = 4
skip_dataset_validation = false
```

Compare four and eight workers on SSD/NVMe. Use one for serial validation on rotational or bandwidth-constrained storage if concurrent reads cause contention. The range is 1..32. Speedup depends on storage bandwidth and hashing throughput; full validation still reads all required bytes.

### Skip full dataset validation at startup

For already verified, unchanged data and caches, merge this field into the existing `[train]` section:

```toml
[train]
skip_dataset_validation = true
```

The default is `false`. This applies to cached image training and image restoration: it skips full cache hashes and record preflight. Restoration also skips HQ/caption reads and content matching, trusting source identities stored in cache headers. Manifests and required headers are still read, with model, count, pairing and batching-shape checks retained. Data readers still check formats when loading samples. Real paired LQ files are still hashed to preserve resume identity; synthetic restoration avoids image content scans entirely. Video/language workflows, cache generation and model-weight hashing are unaffected.

The switch itself does not change resume identity, allowing unchanged data to resume directly. Modified HQ images, captions or cache contents will not be detected at startup in this mode. Disable the option to validate after changing data or rebuilding caches.

### Synthetic super-resolution degradation

The pipeline restores Real-ESRGAN's fixed ordering while preserving the original supervision target:

1. Optional source USM; stage one blur/sinc → resize → noise → compression.
2. Optional stage two blur/sinc → resize → noise.
3. Equal-probability native-LQ delivery: resize + final sinc → compression, or compression → resize + final sinc.
4. Quantize native LQ to 8-bit, then separately bicubic-align to HQ for same-size restoration. No further noise or encoding follows alignment.

```toml
[dataset.groups.synthetic.degradation]
enabled = true
scales = [2.0, 4.0, 6.0, 8.0]
reference_resolution = 400
resolution_exponent = 0.5
second_order_probability = 1.0
use_usm = true
resize_antialias = false
```

Official kernels are fixed pixel distributions, not resolution-calibrated parameters. The official dataset pre-crops to 400 before degradation and then crops training patches to 256. We use 400 as a configurable reference, not a camera calibration. Compute once per sample: `k=clamp((HQ_short_edge/reference_resolution)^resolution_exponent,0.5,2)`. At 256/400/512/1024/1280/2048 this gives about 0.80/1.00/1.13/1.60/1.79/2.00. Apply it to both stages' sigma/support and sinc support/inverse cutoff without recomputing after intermediate resize. Set exponent=0 to disable resolution adaptation. Noise amplitudes, probabilities and codec quality are not spatial quantities and are not scaled.

Reference blur sigma is [0.2,3]/[0.2,1.5], with base support 7..21 and 80% second blur. Each active stage filter substitutes sinc 10% of the time. Final sinc is 80%. Stage cutoff uses π/3..π for small base kernels and π/5..π for sizes >=13; final cutoff is π/3..π. Sinc is untapered. Base resize ranges are [0.15,1.5]/[0.3,1.2].

The project extension `high_scale_protection=true` prevents excessive secondary downsampling beyond x4. With `r=min(1,4/scale)`, second-stage downsample draws become `s'=1-(1-s)*r`: the lower bounds at x4/x6/x8 are 0.30/0.533/0.65. Final sinc uses spatial multiplier `max(0.5,k*r)`. Both blur stages, noise amplitudes and codec qualities remain unchanged; x1..x4 are unaffected. Disable this option as well as resolution adaptation for the original fixed-parameter baseline. Independent operator RNG streams keep geometry and codecs stable when noise categories change.

[HYPIR 2025](https://github.com/XPixelGroup/HYPIR/blob/main/configs/sd2_train.yaml) retains the standard two-stage ranges, 0.1/0.1/0.8 sinc probabilities and resize_back, supporting native LQ followed by alignment. These extensions have not been validated by restoration training metrics.

Each noise pass selects exactly one category: 5% correlated luminance grain, otherwise equally likely Gaussian/Poisson. Gaussian sigma is [1,30]/[1,25], Poisson scale [0.05,3]/[0.05,2.5], gray probability 40%. Poisson follows BasicSR's quantized unique-level convention. Grain is a local extension with sigma [1,8]/255, not a third pass. Two-stage probability of any grain is 9.75%.

Each stage encodes once by default: JPEG/WebP probabilities 85%/15% and 80%/20%. JPEG2000 is supported but disabled by default. JPEG quality is [30,95], WebP [45,98]; extra recompression defaults to zero. Real OpenCV codec round trips are not bit-identical to official GPU DiffJPEG. To select the fixed-parameter baseline, disable both grain probabilities, set both codec probabilities to [1,0,0], and set resolution_exponent=0.

Research and scope: [ResShift](https://github.com/zsyOAOA/ResShift/blob/master/configs/realsr_swinunet_realesrgan256.yaml) provides optional second-order degradation and disabled USM; we expose second_order_probability/use_usm while retaining official defaults 1/true. [BSRGAN](https://github.com/cszn/BSRGAN) explores shuffle and processed sensor noise, deliberately not adopted here. [DiffBIR](https://github.com/XPixelGroup/DiffBIR) and [SUPIR](https://github.com/Fanghua-Yu/SUPIR) motivate target-specific degradation distributions; SUPIR distinguishes light-degradation training, but we do not invent their unreleased training parameters. Optional resize_antialias defaults to false. WebP/grain extensions and spatial adaptation require dataset validation; previews do not establish restoration training gains.

Legacy scenario, noise gain/budget, late noise, independent grain/smear/sharpen fields are removed and rejected explicitly. Migrate configs and recalibrate; the new policy changes resume identity. The direct API accepts trace=[] to report actual grids, kernels, noise parameters and codecs. output_mode="native" is available for direct previews; datasets and calibration require aligned output.

## Editing, references, masks and layers

| Field | Value | Model support |
| --- | --- | --- |
| reference_sources | Ordered reference directories | FLUX Kontext/Fill, FLUX.2/3, applicable Qwen editing/reference variants |
| mask_source | Stem-matched mask directory | FLUX Fill |
| layer_sources | Ordered RGBA foreground-layer directories | Qwen layered |
| control_source | Aligned control directory | Wan |

These are conditions attached to targets, not independent dataset groups. List order determines reference/layer order. Match the [model variant](models.md); unsupported combinations fail before reading weights.

## Video datasets

```toml
[dataset.groups.main]
kind = "video"
source = "/data/videos"
cache_directory = "/data/cache/wan"
resolution = [512, 512]
batch_size = 1

[dataset.groups.main.video]
target_frames = [81]
extraction = "head"
max_frames = 129
target_fps = 16.0
fps_resampling = "fractional"
```

This Wan window recipe uses 1+4k frame counts. head extracts from the beginning; see the [parameter guide](parameters.md) for extraction, stride and samples controls.

Copy H3-specific examples for 24 FPS timestamp resampling, 17n+5 frames, batch_size=1 and native AV JSONL records. Changing family on a generic video-directory recipe does not create valid H3 input.

## When to rebuild caches

| Change | Action |
| --- | --- |
| Images, captions, prompt, references/masks/layers | Rebuild |
| VAE, text encoder, tokenizer, resolution or buckets | Rebuild |
| Only learning rate, update count or output cadence | These fields alone do not require re-encoding; resume still checks numerical identity |
| Restoration training | Keep LQ/HQ after caching; training reads them live |

Every cache invocation fully rebuilds and replaces the old cache only after success. Do not train against a cache being rebuilt.

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
