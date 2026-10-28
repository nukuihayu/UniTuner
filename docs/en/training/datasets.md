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

Synthetic degradation crops the training patch from HQ before creating same-size LQ; it does not resize the entire source into a patch. HQ remains unchanged. For 1536 crops:

```toml
[dataset.groups.main]
resolution = [1536, 1536]
enable_bucket = false

[dataset.groups.main.degradation]
enabled = true
scale_range = [1.0, 8.0]
noise_strength = [0.0, 1.0]
processing_deadzone = 0.1
smear_strength = [0.0, 0.8]
grain_strength = [0.0, 1.5]
grain_scale_range = [1.0, 4.0]
sharpen_strength = [0.0, 0.4]
```

Factors are sampled continuously and uniformly in log space, independently of damage severity, supporting 1..16 including fractional values such as 2.3 and 5.7. With `scale_range=[1,8]`, intervals [1,2], [2,4] and [4,8] have equal probability. Within every factor interval, weak/moderate/strong severity retains approximately 20%/40%/40% mass, without selectable degradation recipes. Explicit `scales=[2.5,4,6,8]` instead samples those candidates uniformly and takes precedence over scale_range. Delivery dimensions are floor(H/factor) and floor(W/factor), so actual factors have pixel-rounding error. A 1536 crop at 4x/6x/8x uses 384/256/192 grids before enlargement back to 1536.

There are no standard/grain/smoothing classes. Zero-based gain ranges use `g(z)=max(0,(z−t)/(1−t))` with `t=processing_deadzone=0.1`. A finite interval maps to exactly zero, joined continuously to positive strengths, without exclusive style switches. At t=0 sampling is purely continuous; at t=1 zero-based gains are disabled. Positive lower bounds explicitly request always-active operations and bypass the deadzone; fixed ranges remain fixed.

Denoising u, independent grain retention v, and global noise n follow Beta(1.5,2). Smoothing uses g(u); grain uses g(v(1−0.5u)); global noise uses g(n). `noise_strength=[0,1]` multiplies acquisition noise, stage-two noise and grain, so zero gain suppresses all added noise. grain_strength is the residual gain before this global multiplier. Sharpening uses a uniform draw through the same deadzone mapping. Zero smoothing also bypasses low-light camera denoising. Defaults put about 7.4% probability at zero global noise and 7.4% at zero smoothing, independently, giving about 0.55% at both. These are theoretical synthetic probabilities, not measured real-photo frequencies. Blur, resizing and compression can remain; this is not a pristine-image guarantee.

Set `noise_strength=[0,0]` to disable all added noise; `smear_strength=[0,0]` disables texture smoothing and camera denoising, while blur settings remain independent. Use `processing_deadzone=0` to retain purely continuous gains, or `noise_strength=[1,1]` to retain full stage-noise amplitude.

| Source scene | Default share | Related processing |
| --- | --- | --- |
| Photo | 25% | Optical blur, resizing, weaker acquisition noise, delivery encoding |
| Web repost | 35% | Intermediate encoding, resizing, stronger delivery compression |
| Low light | 40% | Linear-light shot/read noise, edge-preserving denoising and chroma smoothing |

Source scenes still control acquisition noise and historical encoding tendencies, without selecting exclusive processing branches. Shared severity s uses the smooth CDF `F(s)=0.1s+1.8s²−0.9s³` on [0,1]. Its strictly positive density covers the entire range, placing 20%/40%/40% probability in diagnostic thirds (weak/moderate/strong). One uniform draw is inverted through this CDF, without selecting a class or switching pipelines. Severity s correlates blur, acquisition noise and codec quality; factors are sampled independently. These proportions describe sampled severity, not measured perceptual image damage; source detail, operator switches and independent gains still matter. Explicit fixed factors or scales override the factor distribution. Continuous denoising and residual grain vary separately instead of all damage always increasing together.

Acquisition keeps optical blur first and exchanges noise/resizing. Stage two samples all six blur/noise/resize orders. Delivery uniformly samples every legal ordering of active operators: smoothing precedes sharpening and enlargement; deferred stage-two noise follows enlargement. Grain and encoding can occur on either side. With smoothing, sharpening, grain, encoding and enlargement active there are 40 orders, or 90 with deferred noise. Disabled operators do not bias this distribution. Final sinc filtering and enlargement form one atomic operation: encoding cannot split them. The independent historical encode-then-blur stage remains. Each selected delivery operator runs once; optional codec recompression is separate. These are conditional order counts, not guarantees of coverage in a finite training run.

Correlated physical parameters retain scene severity for 80% of draws; 20% independently explore their full configured ranges. This provides tail and cross-strength coverage without making every combination equally likely. Gain distributions retain their own documented rules; scale is independently log-uniform.

Handheld phone blur uses a straight exposure trajectory with random direction and speed variation, optionally convolved with mild circular defocus. No curved trajectory or geometric warp is used. Defaults: `first.blur.motion_length=[0,10]`, `first.blur.defocus_radius=[0.5,4]`, in 1024-short-edge reference pixels with the same resolution scaling as sigma. Conditional motion/defocus probabilities are 0.20/0.15; with first blur enabled at 0.8, their overall probabilities are 16%/12%. The PSF has unit energy and zero centroid, preserving alignment. This is a global approximation; independently moving subjects, depth-dependent blur, rolling shutter and multi-frame phone ISP are not simulated.

`grain_scale_range=[1,4]` samples the noise-grid reduction relative to the output uniformly in log space. Early grain cannot be finer than a delivery-grid pixel, so enlargement naturally produces coarser grain. Only the noise residual is enlarged and added to the picture; image structure is not resampled to change grain footprint. Grain combines fine luminance and more broadly correlated chroma components, with continuously controlled amplitude.

First-stage blur sigma defaults to [0.4,2.8], scaled by `clamp(sqrt(min(H,W)/1024),0.5,1.5)`: about [0.490,3.429] for square 1536 inputs. Second-stage sigma=[0.15,0.65] uses current-grid pixels. Smoothing uses one moderate edge-preserving pass; `smear_detail_protection=0.6` protects high-contrast contours. Actual downsampling factor d multiplies extra smoothing by `min(1,2/d)`, limiting strong shrinking combined with strong smearing. At 4x/6x/8x, smoothing retains 1/2, 1/3 and 1/4 of its sampled strength. This cannot restore details already lost to downsampling.

`first.noise.exposure=[0.75,1]` lowers linear-light exposure before noise and recovers it afterward. Away from clipping, recovered variance is `signal*shot/exposure + read/exposure²`. Stage-two exposure is [1,1]. This is a processed-RGB approximation, not RAW/demosaicing simulation. Bilinear/bicubic downsampling is antialiased and uses consistent pixel-coordinate rules. No geometric beautification or face warping is introduced.

Encoding uses real JPEG/WebP/JPEG2000 round trips; JPEG/WebP are defaults. JPEG ranges are [40,95] in stage one and [30,95] in stage two. Each codec stage has `recompression_probability=0.25`, shifting codec-block origin before re-encoding and cropping back without moving scene geometry. `compression_blur_probability=0.25` encodes then blurs existing blocks/ringing on the LQ grid using stage-two settings. Zero disables it; explicitly disabling stage-two blur or compression also disables it. Otherwise its probability is independent of the ordinary stage probabilities.


Sinc kernels taper finite support with a Kaiser window (beta=4), retaining mild ringing while reducing artificial outlines amplified by short-kernel normalization. For a 7x7 kernel at cutoff 1.889, a step-response check reduces overshoot from about 24.3% to 3.3%; this is not a universal error bound for images.

Remove old `processing_probabilities` and `grain_smear_probability` settings and use continuous strength ranges. Old keys fail explicitly rather than being ignored. Recalibrate restoration timesteps/sigma and start a new run after distribution changes; the policy version participates in resume identity. Real paired LQ needs no synthetic degradation. Example images validate generation and visual coverage, not improved restoration training or facial fidelity.

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
