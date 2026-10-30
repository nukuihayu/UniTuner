# Models and variants

[简体中文](../../training/models.md) · **English**

[Documentation](../README.md) · [Training](README.md) · [Examples](../examples/README.md)

Ten families share the training workflow while owning native networks, VAEs, text encoding, and conditioning. Every flow-matching entry below supports adapters (LoRA/LoHa/LoKr) and full parameters. Precision follows the base artifact by default; CUDA BF16 AMP is available for flow matching and restoration across all eight image families. Adapter training across all ten families supports [pretraining weight quantization](quantized-base.md) or native quantized bases; full-parameter training requires floating weights.

<details>
<summary>On this page</summary>

- [Choose a model for the task](#choose-a-model-for-the-task)
- [VAE loading](#vae-loading)
- [Model selection](#model-selection)
- [Local assets](#local-assets)
- [Editing, references, and layers](#editing-references-and-layers)
- [Required family-specific settings](#required-family-specific-settings)
- [Video and audio-video](#video-and-audio-video)
- [MOSS-Image-Edit](#moss-image-edit)

</details>

## Choose a model for the task

For subject or style fine-tuning, begin with an image model whose compatible local assets you already have. Editing requires an edit-capable variant and paired reference conditions. Qwen layered uses separate foreground layers. Video/audio-video tasks need their native frame and audio policies in addition to the shared training settings.

Before copying a configuration, identify the exact family, version where applicable, and variant from your checkpoint. Then check its VAE, text encoder, auxiliary bundle, and required dataset conditions. Changing only `model.family` is not a complete model migration.

## VAE loading

Set `model.vae` directly for standard VAE weights; no `weights import` or topology JSON is required. Missing native metadata selects published family defaults. Embedded metadata preserves custom topology, and explicit configurations must still agree with it.

| Family | Default topology and external formats |
| --- | --- |
| Z-Image | Standard 16-channel VAE; detects Diffusers/ComfyUI keys |
| FLUX | Standard 16-channel VAE; detects native/Diffusers keys |
| FLUX.2, FLUX3, MOSS | Shared standard 32-channel packed VAE; detects native/Diffusers keys |
| Qwen Image, Krea2 | Standard 16-channel causal VAE; detects official/ComfyUI keys; Qwen layered selects RGBA by variant |
| Qwen Image 2.1 | Separate 64-channel RGBA VAE; UniTunner's published topology defaults and native keys |
| Wan 2.1/2.2 | Shared 16-channel causal VAE; official `.pth`/`.pt` state dicts and safetensors, with ComfyUI key detection |
| MiniMax-H3 | Built-in video/audio topology; removes video `first_stage_model.`/`video_vae.`/`vae.` and audio `audio_vae.`/`vae.` prefixes |

Loading still validates complete key inventories and tensor shapes, rejecting unknown weights. Custom topology can be supplied through the Python loader's `config` or `weights import --topology`.

## Model selection

| `model.family` | `model.version` | `model.variant` | Inputs and conditions | Restoration | Flow example |
| --- | --- | --- | --- | --- | --- |
| `z-image` | Omit | `base`, `turbo` | Images and Qwen3 text | Yes | [Z-Image](../../examples/flow.toml) |
| `flux` | Omit | `dev`, `kontext`, `fill` | T5 + CLIP; one condition image for Kontext/Fill | Dev only | [Dev](../../examples/flux-flow.toml), [Fill](../../examples/flux-fill-flow.toml) |
| `flux2` | `dev`, `klein-4b`, `klein-9b` | `base`, `distilled`; Dev requires distilled | Native text and optional ordered references | Yes | [FLUX.2](../../examples/flux2-flow.toml), [references](../../examples/flux2-reference-flow.toml) |
| `flux3` | Omit | `base`, `distilled` | Qwen3-VL and up to ten reference sources | Yes | [FLUX3](../../examples/flux3-flow.toml) |
| `moss-image` | Omit | `base`, `sft`, `edit` | Qwen3.5 text and packed VAE | Base/SFT | [MOSS](../../examples/moss-flow.toml) |
| `qwen-image` | Omit | `original`, `edit`, `edit-2509`, `edit-2511`, `layered` | Qwen2.5-VL, edit references or foreground layers | Original only | [Original](../../examples/qwen-image-flow.toml), [edit](../../examples/qwen-image-edit-flow.toml), [layered](../../examples/qwen-image-layered-flow.toml) |
| `qwen-image-2-1` | Omit | `base` | Qwen3-VL, optional references, separate 64-channel VAE | Yes | [Qwen 2.1](../../examples/qwen-image-2-1-flow.toml) |
| `krea2` | Omit | `raw`, `turbo` | Qwen3-VL layerwise text features with native fixed selection | Yes | [Krea2](../../examples/krea2-flow.toml) |
| `wan` | `2.1`, `2.2` | `t2v`, `i2v` | Images/video, native controls and condition frames | No | [Wan](../../examples/wan-flow.toml) |
| `minimax-h3` | Omit | `t2va`, `fl2va`, `ref2va` | Images/audio-video, native AV JSONL records | No | [H3](../../examples/minimax-h3-train.toml) |

Do not set `model.version` for other families. Distilled/Turbo variants retain their model-specific training semantics; base weights, text encoders, and schedules are not interchangeable. Schema validation does not inspect weight topology.

## Local assets

| Field | Purpose |
| --- | --- |
| `model.weights` | Denoiser safetensors, numbered shards, index JSON, or Diffusers directory; accepts the family’s ComfyUI/Diffusers weights; required for training |
| `model.vae` | Native VAE for caching, previews, restoration, and generation |
| `model.text_encoder` | Local encoder directory or native component weights, depending on family |
| `model.assets` | Auxiliary bundle root; defaults to the denoiser file's parent |

Auxiliary bundles use `text_encoder/` for FLUX CLIP and `image_encoder/model.safetensors` for Wan 2.1 I2V CLIP. Encoder weights are never downloaded automatically. Tokenizers/processors have independent Hub defaults, local directory overrides, revisions and offline mode; see [text encoding](text-encoding.md#tokenizer-sources). FLUX `model.text_encoder` is T5. Wan 2.2 I2V does not use the separate 2.1 image encoder. H3 requires its own AV VAE.

`[model.text] max_length` is optional only for families supporting overrides: Z-Image defaults to 512; MOSS to 4096 (maximum 4096); FLUX to 512 (maximum 512); Krea2 to 512. Other families use fixed native budgets and reject explicit overrides. Krea2 does not expose a layer-selection list for the text encoder.

Qwen, Qwen 2.1, Krea2, and H3 detect actual weight namespaces by default; `model.text_source_format` is optional. Supported layouts include native base, Transformers conditional, and Qwen/Comfy language keys under `model.layers.*` with either `visual.*` or `model.visual.*`. Comfy-Org's `qwen3vl_8b_bf16.safetensors` loads directly into a topology-compatible Qwen 2.1 encoder. Explicit `"transformers"`/`"qwen"` overrides remain supported; null/`"auto"` inspect all shard headers. Mixed language namespaces, duplicate visual aliases, missing/extra keys, incompatible shapes and nonfinite values still fail. H3 retains its own width/truncation requirements and validates discarded full-model layers.
FLUX.2/FLUX.3, MOSS and Z-Image also detect supported language namespaces from weights, independently of config architecture declarations. Existing FLUX T5/CLIP alias validation remains; Wan recognizes a uniform `encoder.*` prefix. VAE format/topology validation remains strict; Z-Image now also recognizes a uniform `vae.*` prefix.

## Editing, references, and layers

Conditions belong in the shared `[dataset.groups.main]` section:

```toml
[dataset.groups.main]
source = "/data/targets"
cache_directory = "/data/cache"
reference_sources = ["/data/reference-a", "/data/reference-b"]
```

Each source matches target file stems; list order defines reference order. References are conditions, not additional training examples. Use one entry if only one reference is needed.

Training reference geometry: select a bucket from the target, then resize and center-crop each reference into that same bucket. Aligned equal-sized pairs share scale and crop coordinates; differently composed references are not automatically registered. Upscaling uses Lanczos, downscaling uses area, and resized dimensions use half-up rounding. Qwen 2.1 also supports the explicit independent reference mode described below.

| Model | Reference constraints |
| --- | --- |
| FLUX Dev | Rejects `reference_sources` |
| FLUX Kontext / Fill | Exactly one source; Fill optionally accepts `mask_source` |
| FLUX.2 | Ordered references with the target's bucket geometry |
| FLUX3 | At most ten sources with the target's bucket geometry |
| Qwen edit | Original edit requires references; 2509/2511 can run without them |
| Qwen 2.1 | Ordered references with the target's bucket geometry |
| Qwen original / layered | Reject edit-reference directories; layered uses `layer_sources` |

Qwen 2.1 defaults to paired geometry: target and references convert to RGBA, then share the same resizing, half-up rounding and center-crop operators and target bucket. Equal source dimensions therefore produce equal output dimensions and aligned content. VAE and vision use the same processed reference pixels; only vision composites alpha over white, without another resize.

To enable independent reference geometry, set `reference_resolution = [1024, 1024]` next to `reference_sources` in `[dataset.groups.main]`. This is an area budget (width × height), not a forced square shape. Each reference selects its own 32-aligned aspect bucket and uses the same resampler/crop operator. Portrait references use portrait buckets; alignment can still crop a small border. Omitting the field preserves paired geometry, without padding. This option currently supports Qwen 2.1 only. Changing mode or budget invalidates cache identity; rebuild the cache.

These are training-cache rules. Generation is a separate preprocessing boundary. Rebuild caches after changing preprocessing. Old family-specific aliases such as `dataset.groups.main.flux2.reference_sources` are not used.

For a fixed 1024×1024 target, set `resolution = [1024, 1024]` and `enable_bucket = false`; `reference_resolution = [1024, 1024]` still selects independent reference aspect buckets. Enabling target buckets allows target dimensions to vary with aspect ratio.

For Qwen2.1 original-plus-region-mask conditions, use two ordered `reference_sources` directories and omit `reference_resolution`. Target, original and mask must already share source dimensions and coordinates. The mask is an image reference, not an automatic loss mask. FLUX Fill instead uses native `mask_source`: training and inference share uniform scaling and center cropping, with nearest interpolation for masks. Alpha used as a mask never participates in RGB resampling. Fill target, source and explicit mask must have equal dimensions. Rebuild Fill caches after this fix.

## Required family-specific settings

| Location | Purpose |
| --- | --- |
| `method.guidance` | Learned guidance embedding for FLUX/FLUX.2 Dev, distinct from inference CFG; unavailable in restoration |
| `method.qwen_image.remove_base` | Exclude composite supervision for Qwen layered; default false |
| `dataset.groups.main.wan.single_frame` | Independent condition/target frame indices for Wan I2V image data; mutually exclusive with shared `control_source` |
| `model.minimax_h3` | `audio_vae`, `text_config`, optional `text_processor` / `topology` / `text_device` |
| `dataset.groups.main.minimax_h3` | Cache seed, audio decoding/repair boundaries, and experimental duration policy |
| `method.minimax_h3` | Sigma/focus ranges, `video_shift=12`, `audio_shift=3`, condition cleanliness, `audio_loss_weight=1` |

`dataset.groups.main.control_source` is shared but currently supported only by Wan. Shared field names do not imply support in every family. Old `dataset.groups.main.flux.posterior` / `posterior_seed` fields are not exposed. Flow matching keeps native cache policies; restoration uses shared `method.restoration.posterior`, documented in the [restoration guide](restoration.md).

## Video and audio-video

Wan `dataset.groups.main.video.target_frames` and `max_frames` must follow `1+4k`. The VAE keeps native causal temporal compression. Build [Wan caches](../../examples/wan-cache.toml), then train with the same cache and model. In FP32 mode, native attention still uses BF16 Q/K/V; FP32 parameters do not imply every internal operation is FP32. Explicit CUDA BF16 AMP follows the shared execution policy. Wan 2.1 I2V needs its image encoder; 2.2 keeps its native condition path. A complete FLF2V application is not currently connected.

H3 requires batch size 1 and resolution aligned to 32. Video uses timestamp resampling at 24 FPS, target frame counts `17*n+5`, and `max_frames >= 5`. Native audio is 32 kHz stereo. JSONL is the AV record format, not an image restoration pair format.

Start with [H3 cache](../../examples/minimax-h3-cache.toml), then [H3 training](../../examples/minimax-h3-train.toml). Explicit split caches can be converted with `caches import` and the [import specification](../../examples/minimax-h3-cache-import.json). Cache production does not yet support teacher-caption records; schema readability alone does not make them usable for training.

Video previews and generation use each family's native conditions. See the [example index](../examples/README.md).

[Fill datasets](datasets.md) · [All parameter defaults](parameters.md)

## MOSS-Image-Edit

`model.family="moss-image"` with `model.variant="edit"` reuses MOSS Base/SFT weights and enables the
**Moss-Image-Edit** format in [ComfyUI PR #17296](https://github.com/Comfy-Org/ComfyUI/pull/17296)
(`index_timestep_zero_text_start`). Editing requires training or a matching edit adapter; changing the variant alone does not teach editing.
This is distinct from the PR's two ai-toolkit formats: the text encoder receives instructions only, while reference images enter through packed VAE latents.

References use zero time and attend only to the reference group. Text and target tokens use the current time and attend to all valid tokens.
Reference IDs start at 1, their frame/text coordinate stays at 0, and target coordinates remain after the valid text.
Training adds noise and computes flow loss only on the target, retaining gradients through reference features. Full parameter training and existing adapter algorithms are supported.

- [Training example](../../examples/moss-edit-flow.toml): `source` holds edited targets and matching instruction `.txt` files.
  Ordered `reference_sources` directories hold source images matched by filename stem. Training references use the target bucket size.
- Run `unitunner cache <config> --device cuda`, then `unitunner train <config> --device cuda`.
  Edit caches store ordered reference latents and include reference content in their identity. Original MOSS caches are incompatible.
- [Generation example](../../examples/moss-edit-generate.toml): set `sampling.reference_images`.
  References retain independent sizes aligned to 16 pixels, without FLUX.2 reference resizing. Previews use the same fields.
- Sampling starts from pure target noise. Per-layer reference K/V is cached within one sampling call and shared across steps and CFG branches with matching skipped layers.
  SLG/skipped-layer paths build separate caches, released after sampling and never reused across training updates.
  Edit defaults to no unconditional layer skipping; Base/SFT retain their existing layer-22 default.
- Published adapters retain ComfyUI `diffusion_model.*` keys. Select **Moss-Image-Edit** in the PR's node, not ai-toolkit.
  External repositories' custom adapter files must first be converted to the supported published keys and scaling format.

Edit currently excludes TREAD and restoration and does not lock backgrounds with a mask.
Reference K/V caching trades memory for computation; additional references and skipped-layer paths increase cache memory.

Training retains native MOSS explicit dtype conversions and residual arithmetic; bitwise equality with ComfyUI fused kernels is not promised.

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
