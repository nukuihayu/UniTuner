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

Reference images use their own aspect buckets; equal-sized target/reference pairs share resizing and cropping. Spatially aligned controls and masks use the target bucket. Matching filenames do not register images with different compositions. Check LQ/HQ geometry before paired restoration.

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

The pipeline builds on [HYPIR's official configuration and transform](https://github.com/XPixelGroup/HYPIR/tree/b61d107c6cef38f01a93c7833558869731cfa8c1/HYPIR/dataset), using OMGSR noise/second-blur/JPEG defaults, bounded intermediate resizing, multi-resolution/SR-factor support, multiple codecs and 10% final grain.

1. Crop/paired flip, then use original HQ for both the degradation source and supervised GT: USM is off by default. Explicit `use_usm=true` enables HYPIR USM (fixed support 51, weight 0.5, threshold 10); `sharpen_target=true` also sharpens supervised GT, while false sharpens only the degradation source.
2. Blur/sinc → resize → Gaussian/Poisson → compression.
3. Optional second blur/sinc → resize → Gaussian/Poisson.
4. Equal-probability native [resize+sinc] → compression or compression → [resize+sinc].
5. Bicubic to HQ, one luminance grain pass with 10% probability, then clipping and 8-bit quantization.

```toml
[dataset.groups.synthetic.degradation]
enabled = true
scale_range = [1.0, 8.0]
# scales = [2.0, 4.0, 6.0, 8.0]
reference_resolution = 512
resolution_exponent = 0.5
use_usm = false
sharpen_target = true

[dataset.groups.synthetic.degradation.first.compression]
probabilities = [0.85, 0.10, 0.05]
jpeg_backend = "diffjpeg"

[dataset.groups.synthetic.degradation.second.compression]
probabilities = [0.8, 0.15, 0.05]
jpeg_backend = "diffjpeg"

[dataset.groups.synthetic.degradation.final_grain]
probability = 0.1
sigma = [1.0, 8.0]
correlation = [0.3, 0.8]
```

The two resize ranges are [0.3,1.5] / [0.6,1.2], with up/down/keep probabilities [0.2,0.7,0.1] / [0.3,0.4,0.3]. Every intermediate dimension is bounded below by the final `floor(HQ/scale)` dimension. `scale_range` samples total factors log-uniformly within [1,16]; explicit `scales` takes precedence.

Spatial adaptation is an extension: the HQ coefficient is `k=clamp((HQ_short/reference_resolution)^resolution_exponent,0.5,2)`. On smaller execution grids, blur sigma/support and inverse sinc cutoff use `min(k,max(1,clamp((grid_short/reference_resolution)^resolution_exponent,0.5,2)))`. This limits additional HQ kernel enlargement on small LQ grids while preserving gentle low-resolution HQ adaptation. The coefficient is 1 at 512; exponent 0 retains upstream fixed pixel kernels. USM support stays 51; noise amplitude and JPEG blocks are unchanged. Tiny LQ grids use repeated reflection when Torch reflection padding cannot fit.

Both stages add Gaussian/Poisson with equal probability, 40% gray. Following OMGSR, sigma is [1,15]/[1,12], Poisson amplitude [0.05,2]/[0.05,1], directly on the resized grid with no gain/attenuation. Clip after addition, without rounding. Only Poisson's sampling signal is quantized before counting unique levels.

Mixed blur uses six probabilities [0.45,0.25,0.12,0.03,0.12,0.03], base support 7..21 and sigma [0.2,3]/[0.2,1.5]. Second blur runs 50%; conditional sinc replacement is 10% per active stage, final sinc 80%. Stage cutoff is π/3..π for base support <13 and π/5..π otherwise; final cutoff π/3..π. No taper.

Compression probabilities follow JPEG/WebP/JPEG2000 order: [0.85,0.10,0.05] then [0.8,0.15,0.05]. JPEG uses float DiffJPEG with quality [60,95]/[60,99], 4:2:0, nearest chroma reconstruction and zero padding to multiples of 16. Set `jpeg_backend="opencv"` for real JPEG. WebP quality is [60,98]; JPEG2000 is [250,900]/[200,900]. Real codecs quantize to 8-bit and round quality to an integer. DiffJPEG quality must be strictly between 0 and 100. Recompression is disabled by default.

For JPEG-only degradation, set both compression probabilities to [1,0,0]. Set final grain probability to zero to disable grain.

Final grain has output-pixel RMS [1,8]/255 and correlation radius [0.3,0.8], only once after alignment. Native output skips it. Training/calibration share `degrade_pair`, returning LQ and supervision GT; `degrade_image` is for diagnostics. Trace records actual grids, kernels, noise and codecs. Recalibrate after changing supervision or degradation distributions.


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
