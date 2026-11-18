# Models and variants

[简体中文](../../training/models.md) · English

[Documentation](../README.md) · [Training](README.md) · [Examples](../examples/README.md)

Choose a model for your task, then prepare its matching weights, VAE and text encoder. Flow-matching training in the table supports LoRA, LoHa, LoKr and full parameters. Restoration support depends on the selected variant.

## Choose by task

| Task | Prepare |
| --- | --- |
| Subject or style tuning | An image model, training images and matching captions |
| Image editing | An editing variant, target images, source images and instructions |
| Layered images | Qwen layered, composite targets and separate RGBA foreground layers |
| One-step restoration | A supported model and aligned LQ/HQ pairs or synthetic degradation |
| Video or audio-video | Wan or MiniMaxH3, with matching frame counts, frame rates and audio |

When changing models, check family, variant, version, VAE and text encoder together. Changing the family name alone does not replace the model files.

## Model selection

| `model.family` | `model.version` | `model.variant` | Data and conditions | Restoration | Flow example |
| --- | --- | --- | --- | --- | --- |
| `z-image` | Omit | `base` / `turbo` | Images and text | Yes | [Z-Image](../../examples/flow.toml) |
| `flux` | Omit | `dev` / `kontext` / `fill` | Kontext/Fill need one condition image | Dev only | [FLUX](../../examples/flux-flow.toml), [Fill](../../examples/flux-fill-flow.toml) |
| `flux2` | `dev` / `klein-4b` / `klein-9b` | `base` / `distilled`; Dev requires distilled | Text and optional ordered references | Yes | [FLUX.2](../../examples/flux2-flow.toml), [references](../../examples/flux2-reference-flow.toml) |
| `flux3` | Omit | `base` / `distilled` | Text and up to 10 reference sources | Yes | [FLUX3](../../examples/flux3-flow.toml) |
| `moss-image` | Omit | `base` / `sft` / `edit` | Edit needs references and instructions | Base/SFT | [MOSS](../../examples/moss-flow.toml), [editing](../../examples/moss-edit-flow.toml) |
| `qwen-image` | Omit | `original` / `edit` / `edit-2509` / `edit-2511` / `layered` | Text, edit references or foreground layers | Original only | [Original](../../examples/qwen-image-flow.toml), [edit](../../examples/qwen-image-edit-flow.toml), [layered](../../examples/qwen-image-layered-flow.toml) |
| `qwen-image-2-1` | Omit | `base` | Text and optional references | Yes | [Qwen2.1](../../examples/qwen-image-2-1-flow.toml) |
| `krea2` | Omit | `raw` / `turbo` | Images and text | Yes | [Krea2](../../examples/krea2-flow.toml) |
| `minimax-h3-image` | Omit | `base` | Images, text and up to 16 ordered references | Yes | [H3-Image](../minimax-h3-image.md) |
| `wan` | `2.1` / `2.2` | `t2v` / `i2v` | Images/video; I2V needs condition frames | No | [Wan](../../examples/wan-flow.toml) |
| `minimax-h3` | Omit | `t2va` / `fl2va` / `ref2va` | Images or audio-video records | No | [H3](../../examples/minimax-h3-train.toml) |

Set `model.version` only for FLUX.2 and Wan. Use the weights and sampling settings belonging to the selected Base, Turbo or distilled variant.

## VAE loading

Set `model.vae` to the matching file or directory:

| Model | VAE to prepare |
| --- | --- |
| Z-Image, FLUX | The model's 16-channel VAE |
| FLUX.2, FLUX3, MOSS | The matching FLUX.2 32-channel VAE |
| Qwen Image, Krea2 | The matching 16-channel VAE; Qwen layered needs RGBA support |
| Qwen Image 2.1 | Its 64-channel RGBA VAE |
| Wan | The matching 16-channel VAE in a supported `.pth`, `.pt` or safetensors file |
| MiniMaxH3 | Its video and audio VAEs, configured separately as shown in the H3 example |
| MiniMax-H3-Image | Its image VAE; the PDD schedule and heads are separate model assets |

Standard models normally load their matching files directly. On a loading error, check the model version and component first. See [weight conversion](../weight-conversion.md) when a file needs conversion.

## Local model files

| Field | Value |
| --- | --- |
| `model.weights` | Main model file, shard index or supported model directory |
| `model.vae` | Matching VAE file or directory |
| `model.text_encoder` | Matching text encoder file or directory |
| `model.assets` | Auxiliary-file root; training defaults to the parent of the main weight path |

FLUX's `model.text_encoder` is T5; put CLIP under `model.assets/text_encoder/`. Wan 2.1 I2V uses `image_encoder/model.safetensors`.

Download model weights separately. Tokenizers and processors can use default sources or explicit local paths, revisions and offline settings; see [text encoding](text-encoding.md). Text encoder formats are normally detected automatically. Set `model.text_source_format` only to select a specific supported format.

## Editing, references and layers

Match references to target filename stems:

```toml
[dataset.groups.main]
source = "/data/targets"
cache_directory = "/data/cache"
reference_sources = ["/data/references"]
```

`targets/a.png` pairs with `references/a.png`; `a.txt` contains the instruction. Directory order determines reference order.

| Model | Condition requirements |
| --- | --- |
| FLUX Dev | No references |
| FLUX Kontext/Fill | Exactly one reference directory; Fill can also use `mask_source` |
| FLUX.2 | Optional ordered reference directories |
| FLUX3 | Up to 10 reference directories |
| MOSS edit | At least one reference directory |
| Qwen edit | Original edit needs references; 2509/2511 can omit them |
| Qwen original/layered | No edit references; layered uses `layer_sources` |
| Qwen2.1 | Optional ordered references with a separate area budget |
| MiniMax-H3-Image | Up to 16 ordered references with native image/vision encoding |

Targets and references resize and center-crop using their own aspect ratios. Already aligned images with equal source dimensions keep their pairing; different compositions are not registered automatically. Rerun `cache` after changing images, references or crop settings.

Qwen2.1 can keep square targets and aspect-preserving references:

```toml
[dataset.groups.main]
source = "/data/targets"
cache_directory = "/data/cache"
resolution = [1024, 1024]
enable_bucket = false
reference_sources = ["/data/references"]
reference_resolution = [1024, 1024]
```

`reference_resolution` sets an area budget rather than forcing square references. It is available only for Qwen2.1. To keep a source image and regional mask aligned, use equal original dimensions/coordinates and omit this override. An image reference containing a mask does not become a loss mask automatically.

FLUX Fill's `mask_source` supplies its repainting mask. Keep targets, source images and masks aligned at equal dimensions.

## Model-specific settings

| Setting | Use |
| --- | --- |
| `method.guidance` | FLUX/FLUX.2 Dev training conditioning; distinct from inference CFG |
| `method.qwen_image.remove_base` | Exclude the composite target from Qwen layered supervision |
| `dataset.groups.main.wan.single_frame` | Select Wan I2V condition and target frame indices |
| `model.minimax_h3` | H3 video/audio encoder and processor files |
| `dataset.groups.main.minimax_h3` | H3 cache, audio and duration settings |
| `method.minimax_h3` | H3 video/audio noise and loss settings |

See [parameters](parameters.md) for all fields and [restoration](restoration.md) for posterior and time calibration.

## Video and audio-video

Wan target/max frame counts must follow `1 + 4k`. 2.1 I2V needs its image encoder; configure 2.2 I2V using its own example. Run [Wan cache](../../examples/wan-cache.toml), then [Wan training](../../examples/wan-flow.toml).

H3 requires batch size 1, 32-aligned dimensions, 24 FPS video, target frame counts `17n + 5`, and at least 5 max frames. Audio is 32 kHz stereo. Run [H3 cache](../../examples/minimax-h3-cache.toml), then [H3 training](../../examples/minimax-h3-train.toml).

See [video and audio-video](video.md) for records, condition frames and generation.

## MOSS-Image-Edit

Select `family = "moss-image"` and `variant = "edit"` with matching editing weights or an adapter. Changing the variant alone does not teach a base model to edit.

1. Follow the [editing recipe](../../examples/moss-edit-flow.toml) for targets, matching instructions and reference directories.
2. Run `unituner cache CONFIG --device cuda`, then `unituner train CONFIG --device cuda`.
3. Follow the [generation recipe](../../examples/moss-edit-generate.toml) and set `sampling.reference_images`.
4. Choose **Moss-Image-Edit** in the corresponding ComfyUI node and load a matching adapter.

Use a separate edit cache. More references require more GPU memory. MOSS edit does not support TREAD, one-step restoration or masked background locking.

## Preview defaults

Omit `steps` and `guidance` to use the selected variant's defaults:

| Model | Steps | Guidance |
| --- | --- | --- |
| Qwen Image 2.1 | 25 | 1 |
| Qwen Image / edit / layered | 25 | 4 |
| Z-Image base / turbo | 50 / 9 | 4 / 0 |
| FLUX Dev / Kontext / Fill | 25 | 3.5 / 2.5 / 30 |
| FLUX.2 Dev / Klein base | 50 | 4 |
| FLUX.2 Klein distilled | 4 | 1 |
| FLUX3 base / distilled | 50 / 16 | 4 / no CFG |
| MOSS Image | 50 | 3 |
| Krea2 raw / turbo | 28 / 8 | 5.5 / 1 |

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
