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

Synthetic degradation preserves the HQ framing, brightness and color as supervision while modeling detail loss and processing artifacts in low-resolution photos. Fog, rain, scratches and exposure shifts are not enabled. Prefer sharp, low-noise source photos with minimal compression: existing damage accumulates with synthetic degradation.

| Scene | Default share | Related degradation |
| --- | --- | --- |
| Photo | 55% | Mild optical blur, downsampling, weak noise, optional final encoding |
| Web repost | 30% | Intermediate encoding, resizing, stronger final compression and more frequent mild sharpening |
| Low-light photo | 15% | Linear-light signal-dependent shot/read noise, edge-preserving denoising and chroma smoothing before downsampling/encoding; no target exposure change |

Factors in `scales = [2, 3, 4]` are sampled uniformly, about 33.33% each; `[2, 3, 4, 8]` gives 25% each with no preference for the largest factor. Shared severity follows Beta(2,3), favoring mild/moderate damage while retaining stronger samples. Blur, noise and compression vary slightly around that shared draw. Factor selection does not determine severity. Scene proportions are adjustable training priors, not measured real-world frequencies.

```toml
[dataset.groups.main.degradation]
enabled = true
scales = [2, 3, 4]
# Photo / web repost / low-light; use [0.0, 1.0, 0.0] for web-only data.
scenario_probabilities = [0.55, 0.30, 0.15]
processing_probabilities = [0.65, 0.20, 0.15]
```

Acquisition applies blur, resizing and optional noise, with denoising linked to low-light noise. Only web samples may be encoded at this stage. Delivery resizes to the LQ grid before optional weak blur/noise, then filters, mildly sharpens, optionally encodes and enlarges back to HQ dimensions. Photo/low-light samples have at most one encode; web samples at most two. JPEG/WebP are defaults; JPEG2000 is opt-in.

First-stage `blur.sigma = [0.2, 2.4]` uses a 1024 short-edge reference, multiplied by `clamp(sqrt(min(H,W)/1024), 0.5, 1.5)`: about `[0.245, 2.939]` for square 1536 inputs. This bounded growth is a training prior, not an optical law. Second-stage `[0.15, 0.65]` uses LQ pixels directly and does not grow with HQ dimensions. Centered motion and circular defocus kernels preserve pair alignment. Sinc cutoffs use the local pixel grid; final sinc probability is 0.1 with 7–21 pixel support.

Bilinear/bicubic downsampling is antialiased; default enlargement uses bilinear/bicubic. Mild LQ sharpening has probability 0.3 for photo/low-light and 0.6 for web, with amount 0.1–0.4. See the [parameter reference](parameters.md#degradationconfig) for noise and codec units. Low-light processing approximates a processed RGB pipeline; it is not a complete RAW/demosaicing simulation. Use real paired LQ directly without added synthetic degradation. Recalibrate restoration timesteps after changing distributions; resume configuration identity also changes.

`processing_probabilities = [0.65, 0.20, 0.15]` selects standard processing, residual grain, or texture smoothing followed by sharpening. These paths combine with the source scenes above:

- **Residual grain (20%)**: enlarge the LQ image to at least half the HQ dimensions, add linear-light shot/read noise using `first.noise`, then blend 25% of the denoised result. Encoding and final enlargement follow, preserving grain that would otherwise disappear during SR downsampling. This independent path is not gated by stage `noise_probability`.
- **Texture smoothing + sharpening (15%)**: apply stronger edge-preserving and chroma smoothing on the LQ grid, then sharpen with amount 0.4–0.8. This suppresses skin, hair and fabric microtexture while retaining larger contours, without changing facial geometry.

Use `[1.0, 0.0, 0.0]` to disable the extra processing, `[0.0, 1.0, 0.0]` for grain only, or `[0.0, 0.0, 1.0]` for smoothing only. HQ supervision is untouched. These simulate visible processing artifacts rather than reproducing a specific phone or beautification algorithm.


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
