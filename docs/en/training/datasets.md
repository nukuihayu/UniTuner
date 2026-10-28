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

### Synthetic super-resolution degradation

Synthetic degradation crops the training patch from HQ before creating same-size LQ; it does not resize the entire source into a patch. HQ remains unchanged. For 1536 crops:

```toml
[dataset.groups.main]
resolution = [1536, 1536]
enable_bucket = false

[dataset.groups.main.degradation]
enabled = true
scale_range = [1.0, 8.0]
smear_strength = [0.0, 0.8]
grain_strength = [0.0, 1.5]
grain_scale_range = [1.0, 4.0]
sharpen_strength = [0.0, 0.4]
```

Factors are sampled continuously and uniformly in log space within `scale_range`, supporting 1..16 including fractional values such as 2.3 and 5.7. With [1,8], intervals [1,2], [2,4] and [4,8] have equal probability. Explicit `scales=[2.5,4,6,8]` instead samples those candidates uniformly and takes precedence over scale_range. Delivery dimensions are floor(H/factor) and floor(W/factor), so actual factors have pixel-rounding error. A 1536 crop at 4x/6x/8x uses 384/256/192 grids before enlargement back to 1536.

There are no standard/grain/smoothing classes. Every sample uses the same processing sequence with continuous smoothing, residual grain, grain footprint and sharpening. Denoising u and independent grain retention v follow Beta(1.5,2). Position within the smoothing range is u; position within the grain range is v(1−0.5u). Stronger denoising tends to leave less grain, while both can coexist. Sharpening is uniform within its configured range. These distributions are synthetic priors, not camera calibration. Set a strength range to [0,0] to disable that operation; disabling residual grain does not disable noise in first/second stages.

| Source scene | Default share | Related processing |
| --- | --- | --- |
| Photo | 25% | Optical blur, resizing, weaker acquisition noise, delivery encoding |
| Web repost | 35% | Intermediate encoding, resizing, stronger delivery compression |
| Low light | 40% | Linear-light shot/read noise, edge-preserving denoising and chroma smoothing |

Source scenes still control acquisition noise and historical encoding tendencies, without selecting exclusive processing branches. Shared damage severity follows Beta(2,2); blur, acquisition noise and codec quality vary around it. Continuous denoising and residual grain vary separately instead of all damage always increasing together.

Acquisition begins with optical blur, then randomly orders noise and resizing. Low-light noise is followed by camera denoising; web samples can have intermediate encoding. Stage two randomly orders resizing, local blur and noise. Half of selected stage-two noise is deferred until final enlargement, never added twice. On the delivery grid, optional compression-then-blur and sinc precede continuous content-protected smoothing and sharpening. Final encoding can precede or follow enlargement. Residual grain is added at HQ dimensions; when encoding precedes enlargement, grain represents post-delivery processing residue.

`grain_scale_range=[1,4]` samples the noise-grid reduction relative to the output uniformly in log space. Only the noise residual is enlarged and added to the picture; image structure is not resampled to change grain footprint. Grain combines fine luminance and more broadly correlated chroma components, with continuously controlled amplitude.

First-stage blur sigma defaults to [0.4,2.8], scaled by `clamp(sqrt(min(H,W)/1024),0.5,1.5)`: about [0.490,3.429] for square 1536 inputs. Second-stage sigma=[0.15,0.65] uses current-grid pixels. Smoothing uses one moderate edge-preserving pass; `smear_detail_protection=0.6` protects high-contrast contours. Actual downsampling factor d multiplies extra smoothing by `min(1,2/d)`, limiting strong shrinking combined with strong smearing. At 4x/6x/8x, smoothing retains 1/2, 1/3 and 1/4 of its sampled strength. This cannot restore details already lost to downsampling.

`first.noise.exposure=[0.75,1]` lowers linear-light exposure before noise and recovers it afterward. Away from clipping, recovered variance is `signal*shot/exposure + read/exposure²`. Stage-two exposure is [1,1]. This is a processed-RGB approximation, not RAW/demosaicing simulation. Bilinear/bicubic downsampling is antialiased and uses consistent pixel-coordinate rules. No geometric beautification or face warping is introduced.

Encoding uses real JPEG/WebP/JPEG2000 round trips; JPEG/WebP are defaults. JPEG ranges are [45,95] in stage one and [35,95] in stage two. Each codec stage has `recompression_probability=0.25`, shifting codec-block origin before re-encoding and cropping back without moving scene geometry. `compression_blur_probability=0.25` encodes then blurs existing blocks/ringing on the LQ grid using stage-two settings. Zero disables it; explicitly disabling stage-two blur or compression also disables it. Otherwise its probability is independent of the ordinary stage probabilities.

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
