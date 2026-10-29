# Configuration parameters and filling guide

[简体中文](../../training/parameters.md) · [Training tutorial](README.md)

Defaults below are read from the current configuration schema. A default is not a recommendation for every model. `null` means omit the key: TOML has no null literal. Tables are optional unless their task needs them; required means required within that table. Replace NAME with an ASCII identifier such as main.

For a minimal edit, set paths, model variant, data geometry, learning rate, rank, update count and save cadence. Keep other defaults omitted; detailed examples expose them for tuning.

## Find a setting

| Area | Sections |
| --- | --- |
| Assets and data | [Model](#model) · [Text](#modeltext) · [Groups](#datasetgroupsname) |
| Training | [Updates](#train) · [Adapters](#trainadapter) · [Optimizer](#trainoptimizer) · [Method](#method) |
| Execution and outputs | [Execution](#execution) · [Output](#output) · [Previews](#preview) · [Runtime](#runtime) |
| Restoration | [Method](#methodrestoration) · [Losses](#losses) · [Degradation](#degradationconfig) |
| Auxiliary objectives | [LayerSync](#methodlayersync) · [TREAD](#methodtread) · [REPA](#methodrepa) · [Wavelets](#methodwavelet) |

For command selection and overrides, read [TOML and CLI](../configuration.md). To resolve errors, use [troubleshooting](../troubleshooting.md).

## TOML and CLI values

Use quoted strings, true/false booleans, numeric values and arrays. Paths resolve from the command working directory. On Windows use forward slashes or single-quoted literal paths. Merge edits into an existing table rather than repeating its header.

```toml
[train]
learning_rate = 5e-5
accumulation_steps = 4

[output]
name = "portrait-lora"
# directory = "runs/portrait-lora"
# tensorboard = "runs/tensorboard"
```

```bash
uv run unitunner train train.toml --device cuda --train.learning-rate 5e-5
uv run unitunner train train.toml --device cuda --set 'dataset.groups.main.batch_size=1'
```

## `model`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `family` | "z-image", "moss-image", "flux", "flux2", "flux3", "qwen-image", "qwen-image-2-1", "krea2", "wan", "minimax-h3" | `required` | Model family; match the checkpoint before selecting variant/version. |
| `variant` | string | `required` | Checkpoint variant, such as base/turbo, dev or original; use matching weights. |
| `version` | "dev", "klein-4b", "klein-9b", "2.1", "2.2" / null | `null` | Only FLUX.2 and Wan: dev/klein-4b/klein-9b or 2.1/2.2 respectively. |
| `vae` | string / null | `null` | Matching local VAE weights, required for caching, previews and inference. |
| `text_encoder` | string / null | `null` | Local text encoder file or directory; tokenizer assets are configured separately. |
| `assets` | string / null | `null` | Auxiliary bundle root: FLUX CLIP in text_encoder/, Wan2.1 I2V vision weights in image_encoder/model.safetensors. |
| `text` | object | `table defaults` | Text precision, quantization and tokenizer source table; weights remain in model.text_encoder. |
| `minimax_h3` | object / null | `null` | H3-specific table: encoder assets under model, audio handling under data, schedules/conditioning under method. |
| `text_source_format` | "transformers", "qwen" / null | `null` | Text weight key format transformers/qwen; use only with supported families. |
| `weights` | string | `required` | Local weight file, shard index or supported model directory; replace the example path. |

## `model.text`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `tokenizer` | object | `table defaults` | Use the tokenizer table below; clip_tokenizer is FLUX CLIP, processor is Qwen editing. |
| `clip_tokenizer` | object / null | `null` | Use the tokenizer table below; clip_tokenizer is FLUX CLIP, processor is Qwen editing. |
| `processor` | object / null | `null` | Use the tokenizer table below; clip_tokenizer is FLUX CLIP, processor is Qwen editing. |
| `max_length` | integer / null | `null` | Text token limit; only set for families that support overriding their native budget. |
| `precision` | "fp32", "fp16", "bf16" / null | `null` | Compute precision; training auto follows base weights, CUDA can use bf16, CPU workflows use fp32. |
| `quantization` | "none", "fp8", "convrot-int8", "float8_e4m3fn", "float8_e5m2", "mxfp8", "nvfp4", "int8_tensorwise", "int8_rowwise", "int8_convrot", "convrot_w4a4", "convrot_w4a8", "asym_w4a8_int8", "w6a8_int8" | `"none"` | Storage quantization for frozen base/text weights; none disables it; base quantization requires adapter training. |
| `quantization_rotation_size` | integer；≥ 4 | `256` | Rotation block size for layouts that use rotations; choose a layout-supported dimension. |
| `quantization_group_size` | integer；≥ 4 | `16` | Quantization group size; must satisfy the format's divisibility requirements for the weights. |

## `model.text.tokenizer`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `source` | string / null | `null` | Tokenizer directory or Hub repository ID; omit for the model default. |
| `subfolder` | string / null | `null` | Relative asset subdirectory such as tokenizer; omit when the local path already points directly to its files. |
| `revision` | string / null | `null` | Hub branch/tag/commit; pin a commit for reproducibility and cache it before offline use. |
| `offline` | boolean | `false` | true reads local files/cache only; false allows tokenizer downloads, never automatic encoder-weight downloads. |

## `dataset.defaults`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `batch_size` | integer；> 0 | `1` | Samples per microbatch; reduce this first when activations exceed GPU memory. |
| `repeats` | integer；> 0 | `1` | Repeat each item per data pass; increases a group's relative frequency, not max_updates. |
| `resolution` | [width, height] or [[width, height], ...] | `[1024, 1024]` | One area budget, or samples at every listed scale. Each scale has independent buckets and cache subdirectories. Lists must be nonempty and unique; generation/editing training only for multiple scales. |
| `enable_bucket` | boolean | `true` | Group by aspect ratio; disable for fixed-resolution crops. |
| `bucket_no_upscale` | boolean | `false` | Avoid upscaling small images into larger buckets. |
| `prompt` | string / null | `null` | In a dataset group, overrides every caption sidecar, including with an empty string; in sampling, the generation prompt. |
| `caption_extension` | string / null | `null` | Caption sidecar extension including the dot, e.g. .txt; used when prompt is omitted. |

## `dataset.groups.NAME`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `batch_size` | integer；> 0 | `1` | Samples per microbatch; reduce this first when activations exceed GPU memory. |
| `repeats` | integer；> 0 | `1` | Repeat each item per data pass; increases a group's relative frequency, not max_updates. |
| `resolution` | [width, height] or [[width, height], ...] | `[1024, 1024]` | One area budget, or samples at every listed scale. Each scale has independent buckets and cache subdirectories. Lists must be nonempty and unique; generation/editing training only for multiple scales. |
| `enable_bucket` | boolean | `true` | Group by aspect ratio; disable for fixed-resolution crops. |
| `bucket_no_upscale` | boolean | `false` | Avoid upscaling small images into larger buckets. |
| `prompt` | string / null | `null` | In a dataset group, overrides every caption sidecar, including with an empty string; in sampling, the generation prompt. |
| `caption_extension` | string / null | `null` | Caption sidecar extension including the dot, e.g. .txt; used when prompt is omitted. |
| `reference_sources` | [string, …] | `[]` | Ordered reference directories, stem-matched to targets; only for variants supporting references. |
| `control_source` | string / null | `null` | Aligned Wan control directory; unsupported by other families. |
| `mask_source` | string / null | `null` | FLUX Fill mask directory, stem-matched to targets. |
| `layer_sources` | [string, …] | `[]` | Foreground-layer directories for Qwen layered, stem-matched to composite targets. |
| `kind` | "image", "video", "restoration" | `"image"` | Task/input kind: image/video/restoration for data; flow-matching/restoration for methods. |
| `video` | object / null | `null` | Video window and frame-rate table; add only for video data. |
| `wan` | object / null | `null` | Wan conditioning table; single-frame I2V uses single_frame. |
| `minimax_h3` | object / null | `null` | H3-specific table: encoder assets under model, audio handling under data, schedules/conditioning under method. |
| `source` | string | `required` | Dataset directory or family-specific record file; rebuild caches after editing inputs. |
| `cache_directory` | string | `required` | Dedicated cache directory for this group; never share it with another group. |
| `low_quality_source` | string / null | `null` | LQ restoration directory, stem-matched to HQ source; mutually exclusive with synthetic degradation. |
| `degradation` | object / null | `null` | Two-stage synthetic LQ degradation; mutually exclusive with this group's low_quality_source. |
| `horizontal_flip_probability` | number；≥ 0, ≤ 1 | `0.5` | Paired horizontal-flip probability for restoration, 0..1; use 0 for direction-sensitive tasks. |

## `dataset.groups.NAME.video`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `target_frames` | [integer, …] | `[81]` | Target training frame counts; Wan requires 1+4k, H3 requires 17n+5. |
| `extraction` | "head", "chunk", "slide", "uniform", "full" | `"head"` | Video-window extraction head/chunk/slide/uniform/full; selects windows from each source video. |
| `stride` | integer；> 0 | `1` | Positive stride for video window extraction. |
| `samples` | integer；> 0 | `1` | Window count for uniform video extraction; preview.samples instead holds image sampling tables. |
| `max_frames` | integer；> 0 | `129` | Video processing frame limit; Wan still requires 1+4k and H3 at least 5. |
| `source_fps` | number / null | `null` | Explicit source rate for fractional resampling; omit for timestamps mode. |
| `target_fps` | number；> 0 | `16.0` | Cache target frame rate; H3 requires 24.0. |
| `fps_resampling` | "fractional", "timestamps" | `"fractional"` | fractional uses fractional indices; timestamps uses decoded timestamps and is required by H3. |

## `cache`

| Field | Type / choices | Default | Purpose |
| --- | --- | --- | --- |
| `batch_size` | integer; > 0 | `1` | Encoder batch size, independent of dataset training batch_size. Values above 1 currently support Z-Image only: consecutive same-resolution images share VAE/text forwards, with a smaller final batch. Other families require 1. |

```toml
[cache]
batch_size = 16
```

Alternatively pass `--cache.batch-size 16` to the cache command. Shared training TOML files can retain this table without changing training microbatches. Batch size and peer source contents participate in cache identity because batching can change floating-point rounding; rebuild after changes. Reduce the value if device memory is insufficient.

## `train`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `parameter_mode` | "adapter", "full" | `"adapter"` | adapter trains an adapter; full trains the base and requires removing train.adapter. |
| `adapter` | object / null | `null` | Adapter table: training algorithm under train.adapter; inference weights and config under adapter. |
| `include_layers` | [string, …] | `[]` | Native layer/module/parameter globs; empty selects all eligible targets, not ComfyUI export keys. |
| `exclude_layers` | [string, …] | `[]` | Remove matching targets after include_layers; exclusions take precedence. |
| `learning_rate` | number；≥ 0 | `0.0001` | Base AdamW learning rate; tune against previews and losses while holding data/rank fixed. |
| `weight_decay` | number；≥ 0 | `0.0001` | Decoupled AdamW weight decay; 0 disables it, independently of learning rate. |
| `optimizer` | object | `table defaults` | AdamW implementation and numerical options; defaults suit standard recipes. |
| `max_updates` | integer；> 0 | `1000` | Total effective optimizer updates; accumulated microbatches do not count as separate updates. |
| `accumulation_steps` | integer；> 0 | `1` | Microbatches per optimizer update; trades more work per update for smaller individual batches. |
| `max_grad_norm` | number；≥ 0 | `1.0` | Global gradient-norm clipping threshold; 0 disables clipping. |
| `seed` | integer | `42` | Random seed for recipe comparisons; resume restores RNG from its state file. |
| `skip_dataset_validation` | boolean | `false` | Trust existing image caches at startup and skip full content validation; retain manifest, pairing, model and shape checks. |
| `dataset_validation_workers` | integer; 1..32 | `4` | Concurrent validation workers; 1 is serial. Bounded prefetch preserves sample order and resume identity. |
| `lr_schedule` | "constant", "linear", "cosine" | `"constant"` | constant, linear decay or cosine decay; all support warmup. |
| `warmup_updates` | integer；≥ 0 | `0` | Effective updates for linear LR warmup; must be smaller than max_updates. |

## `train.adapter`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `algorithm` | "lora", "loha", "lokr" | `"lora"` | Adapter algorithm lora/loha/lokr; inference must use the same algorithm. |
| `fused` | boolean | `true` | Fuse eligible CUDA LoRA scaling and residual addition while preserving intermediate rounding; also expose the frozen base and LoRA branch jointly to the compiler. Normally leave unchanged. |
| `rank` | integer；> 0 | `32` | Positive adapter capacity; higher ranks increase trainable parameters, optimizer memory and file size. |
| `alpha` | number；> 0 | `32.0` | Adapter scale; ordinary LoRA uses alpha/rank, commonly initialized with alpha=rank. |

## `train.optimizer`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `betas` | [number, number]；length ≥ 2, length ≤ 2 | `[0.9, 0.999]` | AdamW first/second moment decay factors; each lies in [0,1). |
| `eps` | number；> 0 | `1e-08` | Positive AdamW denominator stabilizer; normally keep 1e-8. |
| `backend` | "auto", "fused", "foreach", "single" | `"auto"` | auto chooses the optimizer implementation; fused/foreach/single select it explicitly. |

## `method`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `kind` | "flow-matching", "restoration" | `"flow-matching"` | Task/input kind: image/video/restoration for data; flow-matching/restoration for methods. |
| `layersync` | object / null | `null` | Flow-matching layer self-supervision with a weight and student/teacher indices. |
| `repa` | object / null | `null` | Flow feature alignment requiring a feature encoder, actual digest and separate feature cache. |
| `tread` | object / null | `null` | Flow token routing with a drop fraction and eligible block interval. |
| `wavelet` | object / null | `null` | Wavelet objective table; its wavelet field selects haar/db4, with band and time weights below. |
| `guidance` | number / null | `null` | Learned guidance embedding for supported FLUX variants; not inference CFG. |
| `restoration` | object / null | `null` | Restoration time, VAE encoder adapter and feature-model table; only for restoration. |
| `qwen_image` | object / null | `null` | Qwen layered supervision options; omit for other models. |
| `minimax_h3` | object / null | `null` | H3-specific table: encoder assets under model, audio handling under data, schedules/conditioning under method. |

## `method.restoration`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `midtimestep` | integer；≥ 1, ≤ 998 | `244` | Restoration integer time 1..998, default 244; its sigma depends on the model's schedule. |
| `sigma` | number / null | `null` | Restoration noise level strictly within (0,1); do not enable calibrate simultaneously. |
| `calibrate` | boolean | `false` | Estimate restoration sigma from data before training; keep false for a fixed-time recipe. |
| `calibration_pairs` | integer；> 0 | `32` | Number of pairs for automatic calibration; a nondefault value requires calibrate=true. |
| `encoder_rank` | integer；≥ 0 | `4` | VAE encoder adapter rank; 0 disables it and removes the need for encoder_weights at inference. |
| `posterior` | "mean", "sample" / null | `null` | Training VAE posterior mean/sample; omission uses the family's policy, inference/previews use means. |
| `histogram_bins` | integer；≥ 2 | `32` | Color histogram bin count, at least 2; relevant when histogram loss is enabled. |
| `dino_weights` | string / null | `null` | Local DINOv3 ConvNeXt weights, required when dv3d or gan is positive. |
| `vgg_weights` | string / null | `null` | Local VGG19 weights, required when fdl is positive. |
| `feature_precision` | "bf16", "fp32" | `"bf16"` | Perceptual-network precision bf16/fp32, independent of the main model compute dtype. |
| `dino_widths` | [integer, …]；length ≥ 4, length ≤ 4 | `[192, 384, 768, 1536]` | Four DINO ConvNeXt stage widths; defaults match Large, change only for a matching alternate topology. |
| `dino_depths` | [integer, …]；length ≥ 4, length ≤ 4 | `[3, 3, 27, 3]` | Four DINO stage depths; must match the selected checkpoint. |
| `vgg_widths` | [integer, …]；length ≥ 5, length ≤ 5 | `[64, 128, 256, 512, 512]` | Five VGG stage widths; must match the selected checkpoint. |

## `losses`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `lrr` | number；≥ 0 | `5.0` | Latent restoration regression coefficient; 0 disables it. |
| `dv3d` | number；≥ 0 | `3.0` | DINOv3 feature-distance coefficient; positive values require DINO weights. |
| `l1` | number；≥ 0 | `0.5` | Pixel L1 coefficient between output and HQ; 0 disables it. |
| `gan` | number；≥ 0 | `0.5` | Generator adversarial coefficient; positive values enable a discriminator and require DINO weights. |
| `fdl` | number；≥ 0 | `0.5` | VGG frequency-distribution coefficient; positive values require VGG weights. |
| `histogram` | number；≥ 0 | `0.0` | Color histogram loss coefficient; needs no additional feature model. |

## `execution`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `precision` | "auto", "fp32", "fp16", "bf16" | `"auto"` | Compute precision; training auto follows base weights, CUDA can use bf16, CPU workflows use fp32. |
| `fp16_loss_scale` | number；≥ 1.0 | `128.0` | FP16 loss-scaling factor; used only by the FP16 path. |
| `compile` | boolean / "auto" | `false` | false uses eager execution; true compiles on CUDA, auto selects by device; compilation adds startup cost. |
| `recompute` | "none", "block" | `"none"` | none retains activations; block recomputes blocks to reduce activation memory at extra compute cost. |
| `recompute_interval` | integer；> 0 | `1` | Block recomputation interval; keep 1 when the offload policy requires every block to be recomputed. |
| `restoration_recompute` | "inherit", "none", "block" | `"inherit"` | Restoration VAE and DINO recomputation. inherit follows recompute; block keeps auxiliary networks checkpointed independently of recompute_interval while reducing DiT recomputation. |
| `attention_activation_offload` | boolean | `false` | Offload saved attention activations to host memory at extra transfer and RAM cost. |
| `residency` | "resident", "frozen-blocks", "vbar", "trainable-blocks" | `"resident"` | resident keeps weights on device; frozen-blocks/vbar manage frozen weights; trainable-blocks is reserved and currently unsupported. |
| `use_pinned_memory_for_block_swap` | boolean / null | `null` | Pinned host memory for block swapping; omitted selects the policy default and can increase host RAM use. |
| `frozen_weight_cache_mb` | integer / null | `null` | Frozen-block CUDA cache budget in MiB; omitted means automatic, 0 minimizes retention but still needs workspace. |
| `quantization` | "none", "fp8", "convrot-int8", "float8_e4m3fn", "float8_e5m2", "mxfp8", "nvfp4", "int8_tensorwise", "int8_rowwise", "int8_convrot", "convrot_w4a4", "convrot_w4a8", "asym_w4a8_int8", "w6a8_int8" | `"none"` | Storage quantization for frozen base/text weights; none disables it; base quantization requires adapter training. |
| `quantization_rotation_size` | integer；≥ 4 | `256` | Rotation block size for layouts that use rotations; choose a layout-supported dimension. |
| `quantization_group_size` | integer；> 0 | `16` | Quantization group size; must satisfy the format's divisibility requirements for the weights. |

## `output`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `directory` | string | `"runs"` | Output directory; omit for runs/<name>_<timestamp>, set an explicit path to reuse it. |
| `name` | string | `"model"` | Weight filename prefix and the basis for log/TensorBoard run names. |
| `log_every_seconds` | number；> 0 | `60.0` | Noninteractive summary interval, default 60 seconds; terminals refresh in place and files retain every update. |
| `save_every_updates` | integer；> 0 | `100` | Export weights and independent resume state every N effective updates, plus the final update. |
| `max_resume_checkpoints` | integer；≥ 0 | `1` | Keep the newest N resume snapshots; default 1, 0 keeps all. |
| `max_weight_checkpoints` | integer；≥ 0 | `0` | Keep the newest N weight sets; default 0 keeps all, including companion encoders. |
| `tensorboard` | string / null | `null` | Event root directory; the trainer automatically creates a run subdirectory from the training name. |
| `save_precision` | "fp32", "fp16", "bf16" | `"fp32"` | Export dtype fp32/bf16/fp16 for inference weights; resume retains training dtypes. |
| `checkpoint_storage` | "complete", "frozen-reference" | `"complete"` | complete includes full training tensors; frozen-reference omits duplicate frozen parameters and requires the identical base. |

## `preview`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `every_updates` | integer；> 0 | `100` | Preview every N effective updates and the final update; affects observation cadence only. |
| `at_start` | boolean | `false` | Render an initial baseline for fresh runs; resume does not replay it. |
| `samples` | [object, …] | `[]` | Window count for uniform video extraction; preview.samples instead holds image sampling tables. |
| `videos` | [object, …] | `[]` | Array of Wan video preview sampling tables. |
| `audio_video` | [object, …] | `[]` | Array of H3 audio/video preview tables containing inputs and sampling. |
| `num_images` | integer；> 0 | `1` | Number of restoration preview items per round, rotating in a seed-fixed order. |

## `preview.samples`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `prompt` | string | `""` | Prompt used for inference or the preview. |
| `negative_prompt` | string / null | `null` | Sampling negative prompt; support and CFG semantics depend on the model variant. |
| `reference_images` | [string, …] | `[]` | Ordered inference reference-image paths; follow the selected variant's requirements. |
| `conditioning_image` | string / null | `null` | Single conditioning image path, e.g. FLUX Kontext/Fill or Wan I2V; follow family constraints. |
| `mask_image` | string / null | `null` | Inference mask image for variants supporting inpainting. |
| `condition_posterior` | "sample", "mean" | `"sample"` | FLUX conditioning-image VAE posterior sample/mean; not a general override for all models. |
| `conditioning_resize` | "preferred", "original", "output" | `"preferred"` | Inference reference resize policy preferred/original/output; output is supported only by applicable Qwen models. |
| `output_layers` | integer / null | `null` | Foreground layer count for Qwen layered only. |
| `caption_language` | "en", "zh" / null | `null` | Layer-caption language en/zh for Qwen layered. |
| `width` | integer；≥ 16 | `1024` | Output width in pixels; generally align to 16, or 32 for Qwen2.1/H3. |
| `height` | integer；≥ 16 | `1024` | Output height in pixels; together with width determines sampling memory use. |
| `seed` | integer | `42` | Sampling noise seed; keep fixed to compare checkpoints. |
| `attention_backend` | "torch", "kitchen-int8" | `"torch"` | torch uses native attention; kitchen-int8 requires the relevant backend and supported inference paths. |
| `steps` | integer / null | `null` | Sampling iterations; omission uses variant defaults, do not reuse one value across Turbo and nondistilled models. |
| `shift` | number / null | `null` | Sampling schedule shift; interpretation is model-specific, omit for native defaults. |
| `guidance` | number / null | `null` | Native model sampling guidance/CFG; omit for variant defaults. |
| `training_shift` | number；> 0 | `1.0` | Sampling multiplier matching MOSS training schedule shift; do not set for unrelated families. |
| `uncond_skip_layers` | [integer, …] / null | `null` | Zero-based blocks skipped by the MOSS unconditional branch; match actual model depth. |
| `slg_layers` | [integer, …] | `[]` | MOSS SLG skipped-block list; only set for supported models. |
| `slg_scale` | number；≥ 0 | `0.0` | SLG guidance strength; 0 disables it. |
| `slg_start` | number；≥ 0, ≤ 1 | `0.0` | Sampling progress fraction where SLG starts, 0..1. |
| `slg_stop` | number；≥ 0, ≤ 1 | `1.0` | Sampling progress where SLG stops; must be at least slg_start. |

## `runtime`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `cuda_allow_tf32` | boolean | `false` | Allow TF32 matrix operations on supported GPUs; may change numerical trajectories. |
| `cudnn_enabled` | boolean | `true` | Enable cuDNN convolution kernels; normally keep true. |
| `cudnn_benchmark` | boolean | `true` | Benchmark convolution algorithms by default; startup and new shapes incur search overhead. Can be disabled explicitly. |
| `cudnn_deterministic` | boolean | `false` | Restrict cuDNN to deterministic algorithms for reproducibility, potentially reducing speed. |

## `method.layersync`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `weight` | number；> 0 | `1.0` | Coefficient of this auxiliary objective in the total loss; 0 removes its contribution. |
| `student_layer` | integer；≥ 0 | `0` | Zero-based native student block index for LayerSync. |
| `teacher_layer` | integer；≥ 0 | `1` | Zero-based native teacher block index for LayerSync. |

## `method.tread`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `drop_fraction` | number；> 0, < 1 | `0.5` | TREAD token-drop fraction within the selected block range; start conservatively. |
| `start_layer` | integer；≥ 0 | `0` | Zero-based first block index for TREAD. |
| `end_layer` | integer；> 0 | `1` | TREAD ending block boundary; obey method and model range constraints. |

## `method.repa`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `weight` | number；> 0 | `1.0` | Coefficient of this auxiliary objective in the total loss; 0 removes its contribution. |
| `layer` | integer；≥ 0 | `0` | Zero-based native block index for REPA feature alignment. |
| `encoder_dim` | integer；> 0 | `required` | REPA target feature dimension; must match the selected feature encoder. |
| `encoder_sha256` | string；pattern ^[0-9a-f]{64}$ | `required` | Actual SHA256 of the REPA encoder file; replace the all-zero placeholder. |
| `cache_directory` | string | `required` | Dedicated cache directory for this group; never share it with another group. |
| `encoder_weights` | string / null | `null` | Restoration encoder adapter or REPA feature-encoder path, according to the containing table. |
| `encoder_device` | string | `"cpu"` | REPA feature encoder device cpu/cuda; affects caching memory use. |

## `method.wavelet`

Weighted DWT band MSE. In restoration this compares decoded RGB output to detached HQ, with no timestep decay; see [restoration](restoration.md).

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `weight` | number；> 0 | `1.0` | Coefficient of this auxiliary objective in the total loss; 0 removes its contribution. |
| `wavelet` | "haar", "db4" | `"db4"` | Wavelet objective table; its wavelet field selects haar/db4, with band and time weights below. |
| `levels` | integer；≥ 1, ≤ 8 | `3` | Wavelet decomposition levels, 1..8; deeper levels include lower spatial frequencies. |
| `ll` | number；≥ 0 | `0.1` | Loss coefficient for the low-frequency LL wavelet band. |
| `lh` | number；≥ 0 | `0.01` | Loss coefficient for the LH detail wavelet band. |
| `hl` | number；≥ 0 | `0.01` | Loss coefficient for the HL detail wavelet band. |
| `hh` | number；≥ 0 | `0.05` | Loss coefficient for the high-frequency HH wavelet band. |
| `ll_level_threshold` | integer / null | `-1` | Decomposition-level threshold for LL loss; -1 uses the implementation's default selection. |
| `normalize_bands` | boolean | `false` | Normalize wavelet band loss scales; keep consistent between comparisons. |
| `timestep_cutoff` | number；≥ 0, ≤ 1 | `0.7` | Time-weight cutoff for the wavelet objective, 0..1. |
| `timestep_transition_width` | number；> 0 | `0.4` | Positive smoothing width around the wavelet time-weight transition. |

## `method.qwen_image`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `remove_base` | boolean | `false` | Whether Qwen layered excludes composite/base supervision; only for layered flow training. |

## `model.minimax_h3`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `audio_vae` | string | `required` | Local H3 audio VAE weights. |
| `text_config` | string | `required` | H3 text encoder configuration matching its weights. |
| `text_processor` | string / null | `null` | Local H3 processor directory; otherwise the family default source can be used. |
| `text_device` | "cpu", "cuda" / null | `null` | H3 text encoding device; set when placing the encoder separately. |
| `video_topology` | object / null | `null` | H3 nonstandard video VAE topology override; normally omit for standard weights. |
| `audio_topology` | object / null | `null` | H3 nonstandard audio VAE topology override; normally omit for standard weights. |

## `method.minimax_h3`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `sigma_min` | number；≥ 0, ≤ 1 | `0.0` | Lower H3 training sigma sampling bound. |
| `sigma_max` | number；≥ 0, ≤ 1 | `1.0` | Upper H3 training sigma sampling bound. |
| `focus_min` | number；≥ 0, ≤ 1 | `0.0` | Lower H3 focused-time interval bound. |
| `focus_max` | number；≥ 0, ≤ 1 | `1.0` | Upper H3 focused-time interval bound. |
| `focus_probability` | number；≥ 0, ≤ 1 | `0.0` | Probability of sampling the H3 focused-time interval. |
| `video_shift` | number；≥ 0.01, ≤ 100 | `12.0` | H3 video schedule shift, default 12.0; match the training/sampling recipe. |
| `audio_shift` | number；≥ 0.01, ≤ 100 | `3.0` | H3 audio schedule shift, default 3.0. |
| `visual_condition_clean` | number；≥ 0, ≤ 1 | `0.999` | H3 visual-condition cleanliness; default 0.999. |
| `audio_condition_clean` | number；≥ 0, ≤ 1 | `1.0` | H3 audio-condition cleanliness; default 1.0. |
| `audio_loss_weight` | number；≥ 0 | `1.0` | Relative H3 audio loss coefficient; default 1.0. |

## `dataset.groups.NAME.wan.single_frame`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `clean_sources` | [string, …]；length ≥ 1, length ≤ 2 | `required` | Clean conditioning-frame directories for Wan single-frame I2V. |
| `clean_indices` | [integer, …]；length ≥ 1, length ≤ 2 | `required` | Condition-frame indices corresponding to clean_sources in order. |
| `target_index` | integer；≥ 0, < 1024 | `required` | Target frame index; match the record and conditioning indices. |

## `dataset.groups.NAME.minimax_h3`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `cache_seed` | integer | `0` | H3 cache-generation seed for repeatable random conditions. |
| `allow_experimental_duration` | boolean | `false` | Allow experimental H3 durations; keep false for standard recipes. |
| `audio_decode` | object | `table defaults` | H3 audio decoding rate, channel and timestamp-continuity table. |
| `audio_pad_tolerance` | integer；≥ 0 | `800` | H3 audio alignment padding tolerance in native units and bounds. |
| `audio_max_repair_samples` | integer；≥ 0 | `6400` | Maximum H3 audio samples repaired for alignment, not training item count. |

## `dataset.groups.NAME.minimax_h3.audio_decode`

| Parameter | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `sample_rate` | integer；> 0 | `required` | Decoded audio sample rate; native H3 uses 32000 Hz. |
| `channels` | integer；≥ 1, ≤ 2 | `required` | Decoded audio channel count; native H3 uses stereo. |
| `max_discontinuity_seconds` | number；≥ 0 | `0.05` | Allowed audio timestamp discontinuity in seconds. |

## `DegradationConfig`

HYPIR stage ordering and USM supervision; final grain is a 10% post-alignment extension. See [datasets](datasets.md). 旧阶段 grain 字段迁移到 final_grain。

| Field / 字段 | Default / 默认 | Meaning / 说明 |
| --- | --- | --- |
| `enabled` | `false` | Enable synthetic degradation / 启用合成退化 |
| `scales / scale_range` | `null / [1,8]` | Explicit factors override log-uniform range, 1..16 / 离散倍率优先 |
| `reference_resolution / resolution_exponent` | `512 / 0.5` | Single HQ spatial scale clamp((r/ref)^exponent,0.5,2); exponent=0 disables adaptation / HQ 单次空间适配 |
| `high_scale_protection` | `true` | Above x4, narrow second downsampling and reduce final sinc footprint; false restores baseline / x4 以上限制过度缩小和末端低通 |
| `noise_grid_adaptation` | `true` | Scale acquisition residual by min(1,sqrt(current_area/output_area)) before clipping; final grain unaffected / 按实际执行网格抑制粗颗粒 |
| `second_order_probability` | `1.0` | Probability of running stage two; final delivery always runs / 第二阶段概率 |
| `use_usm / sharpen_target` | `false / true` | USM disabled; sharpen_target applies only when enabled / 默认关闭 USM，启用后才应用目标锐化设置 |
| `final_grain` | `subtable` | GrainConfig after alignment / 回原尺寸后的 grain |
| `resize_antialias` | `false` | Optional antialias for bilinear/bicubic downsampling / 可选抗混叠 |
| `output_mode` | `aligned` | aligned or native; datasets/calibration require aligned / 原生 LQ 仅用于直接调用 |
| `resize_modes` | `["area","bilinear","bicubic"]` | Random stage interpolation; nearest is opt-in / 阶段插值候选 |
| `first / second` | `subtable` | StageConfig / SecondStageConfig |
| `final_sinc_probability` | `0.8` | Final sinc on native LQ, before alignment / 对齐放大前的 LQ sinc |
| `gray_probability / color_jitter_probability` | `0 / 0` | Optional output augmentation / 输出增强 |
| `brightness / contrast / saturation / hue` | `[0.5,1.5] / [0.5,1.5] / [0,1.5] / [-0.1,0.1]` | Ranges for enabled color jitter / 已启用颜色增强的范围 |

## `StageConfig`

| Field / 字段 | Default / 默认 | Meaning / 说明 |
| --- | --- | --- |
| `blur / noise / compression` | `subtable` | BlurConfig / NoiseConfig / CompressionConfig |
| `blur_probability / sinc_probability` | `1.0 / 0.1` | Stage filter probability / conditional sinc replacement probability / 阶段滤波及条件 sinc 概率 |
| `resize_probabilities / resize_range` | `[0.2,0.7,0.1] / [0.15,1.5]` | Up/down/keep, unrestricted by SR factor / 升降保持，不按总倍率裁限 |
| `noise_probability / compression_probability` | `1.0 / 1.0` | Each runs once at its fixed position / 固定顺序各一次 |

## `SecondStageConfig`

| Field / 字段 | Default / 默认 | Meaning / 说明 |
| --- | --- | --- |
| `blur / noise / compression` | `subtable` | SecondBlurConfig / SecondNoiseConfig / SecondCompressionConfig |
| `blur_probability / sinc_probability` | `0.8 / 0.1` | No scale-dependent probability attenuation / 不随倍率衰减概率 |
| `resize_probabilities / resize_range` | `[0.3,0.4,0.3] / [0.3,1.2]` | Relative to HQ/scale / 相对最终 LQ |
| `noise_probability / compression_probability` | `1.0 / 1.0` | Final compression exchanges with native resize+sinc / 末端压缩只与 resize+sinc 整组交换 |

## `BlurConfig`

| Field / 字段 | Default / 默认 | Meaning / 说明 |
| --- | --- | --- |
| `sigma` | `[0.2,3.0]` | Uniform reference-pixel sigma, multiplied once by HQ spatial scale / 独立均匀 sigma |
| `probabilities` | `[0.45,0.25,0.12,0.03,0.12,0.03]` | Isotropic/anisotropic Gaussian, generalized, plateau / 各向同性与异性六类 |
| `generalized_beta / plateau_beta` | `[0.5,4] / [1,2]` | Official beta distributions / 官方 beta 分布 |

## `SecondBlurConfig`

| Field / 字段 | Default / 默认 | Meaning / 说明 |
| --- | --- | --- |
| `sigma` | `[0.2,1.5]` | All other settings inherited from BlurConfig / 其余继承 |

## `NoiseConfig`

| Field / 字段 | Default / 默认 | Meaning / 说明 |
| --- | --- | --- |
| `gaussian_probability` | `0.5` | Gaussian vs Poisson / 两种采集噪声的选择概率 |
| `gaussian_sigma / poisson_scale` | `[1,30] / [0.05,3]` | Official standard deviation (8-bit units) / Poisson amplitude / 官方幅度 |
| `gray_probability` | `0.4` | Gray vs color Gaussian/Poisson / 灰度噪声概率 |

## `SecondNoiseConfig`

| Field / 字段 | Default / 默认 | Meaning / 说明 |
| --- | --- | --- |
| `gaussian_sigma / poisson_scale` | `[1,25] / [0.05,2.5]` | Other fields inherited / 其余继承 |

## `GrainConfig`

| Field / 字段 | Default / 默认 | Meaning / 说明 |
| --- | --- | --- |
| `probability` | `0.1` | One post-alignment draw; skipped for native / 回原尺寸后只抽一次 |
| `sigma / correlation` | `[1,8] / [0.3,0.8]` | 8-bit RMS and output-pixel radius, no spatial multiplier / 不随倍率或分辨率放大 |

## `CompressionConfig`

| Field / 字段 | Default / 默认 | Meaning / 说明 |
| --- | --- | --- |
| `probabilities` | `[0.85,0.15,0]` | JPEG / WebP / JPEG2000 |
| `jpeg_quality / webp_quality / jpeg2000_quality` | `[30,95] / [45,98] / [250,900]` | Uniform codec-native ranges / 编码器质量区间 |
| `recompression_probability` | `0` | Optional extra encoding, disabled in baseline / 默认关闭额外重编码 |

## `SecondCompressionConfig`

| Field / 字段 | Default / 默认 | Meaning / 说明 |
| --- | --- | --- |
| `probabilities` | `[0.8,0.2,0]` | JPEG / WebP / JPEG2000 |
| `jpeg2000_quality` | `[200,900]` | Other quality fields inherited / 其余质量区间继承 |

## `restore` command

Use a separate TOML and the matching command. A sampling table uses the same image-sampling fields listed under preview.samples where applicable.

| Field | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `runtime` | object | `subtable defaults` | Shared CUDA/cuDNN backend options, configurable per command. |
| `model` | object | `required` | Model selection and local assets; calibration needs only family and vae. |
| `input` | string | `required` | Input restoration image; output preserves its oriented dimensions. |
| `output` | string | `required` | Inference output file path; usually PNG for images, family-specific for video. |
| `prompt` | string | `""` | Prompt used for inference or the preview. |
| `attention_backend` | "torch", "kitchen-int8" | `"torch"` | torch uses native attention; kitchen-int8 requires the relevant backend and supported inference paths. |
| `sigma` | number / null | `null` | Restoration noise level strictly within (0,1); do not enable calibrate simultaneously. |
| `adapter` | object / null | `null` | Adapter table: training algorithm under train.adapter; inference weights and config under adapter. |
| `encoder_weights` | string / null | `null` | Restoration encoder adapter or REPA feature-encoder path, according to the containing table. |

## `generate` command

Use a separate TOML and the matching command. A sampling table uses the same image-sampling fields listed under preview.samples where applicable.

| Field | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `runtime` | object | `subtable defaults` | Shared CUDA/cuDNN backend options, configurable per command. |
| `model` | object | `required` | Model selection and local assets; calibration needs only family and vae. |
| `sampling` | object | `subtable defaults` | Inference sampling table; choose image, Wan or H3 fields as appropriate. |
| `adapter` | object / null | `null` | Adapter table: training algorithm under train.adapter; inference weights and config under adapter. |
| `output` | string | `required` | Inference output file path; usually PNG for images, family-specific for video. |

## `Wan generate-video` command

Use a separate TOML and the matching command. A sampling table uses the same image-sampling fields listed under preview.samples where applicable.

| Field | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `runtime` | object | `subtable defaults` | Shared CUDA/cuDNN backend options, configurable per command. |
| `model` | object | `required` | Model selection and local assets; calibration needs only family and vae. |
| `sampling` | object | `subtable defaults` | Inference sampling table; choose image, Wan or H3 fields as appropriate. |
| `adapter` | object / null | `null` | Adapter table: training algorithm under train.adapter; inference weights and config under adapter. |
| `high_noise` | object / null | `null` | Wan2.2 high-noise model/adapter settings; follow its video-generation example. |
| `output` | string | `required` | Inference output file path; usually PNG for images, family-specific for video. |

## `H3 generate-video` command

Use a separate TOML and the matching command. A sampling table uses the same image-sampling fields listed under preview.samples where applicable.

| Field | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `inputs` | object | `required` | H3 prompt, first/last frame, control and reference input table. |
| `sampling` | object | `subtable defaults` | Inference sampling table; choose image, Wan or H3 fields as appropriate. |
| `runtime` | object | `subtable defaults` | Shared CUDA/cuDNN backend options, configurable per command. |
| `model` | object | `required` | Model selection and local assets; calibration needs only family and vae. |
| `adapter` | object / null | `null` | Adapter table: training algorithm under train.adapter; inference weights and config under adapter. |
| `output` | string | `required` | Inference output file path; usually PNG for images, family-specific for video. |

## `mid-timestep` command

Use a separate TOML and the matching command. A sampling table uses the same image-sampling fields listed under preview.samples where applicable.

| Field | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `runtime` | object | `subtable defaults` | Shared CUDA/cuDNN backend options, configurable per command. |
| `model` | object | `required` | Model selection and local assets; calibration needs only family and vae. |
| `dataset` | object | `required` | Named data groups and optional defaults; each group declares its source. |
| `seed` | integer | `42` | Random seed for recipe comparisons; resume restores RNG from its state file. |
| `posterior` | "mean", "sample" / null | `null` | Training VAE posterior mean/sample; omission uses the family's policy, inference/previews use means. |
| `calibration_pairs` | integer | `32` | Number of pairs for automatic calibration; a nondefault value requires calibrate=true. |

## `wan-generate.sampling`

| Field | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `prompt` | string | `""` | In a dataset group, overrides every caption sidecar, including with an empty string; in sampling, the generation prompt. |
| `negative_prompt` | string | `""` | Sampling negative prompt; support and CFG semantics depend on the model variant. |
| `width` | integer | `832` | Output width in pixels; generally align to 16, or 32 for Qwen2.1/H3. |
| `height` | integer | `480` | Output height in pixels; together with width determines sampling memory use. |
| `frames` | integer | `81` | Generated frame count; obey family constraints, with fps determining duration. |
| `fps` | integer | `16` | Generation frame rate; H3 uses native 24 FPS. |
| `steps` | integer | `40` | Sampling iterations; omission uses variant defaults, do not reuse one value across Turbo and nondistilled models. |
| `solver` | "euler", "dpm++", "unipc" | `"unipc"` | Wan sampling solver; choose a family-supported value. |
| `shift` | number | `5.0` | Sampling schedule shift; interpretation is model-specific, omit for native defaults. |
| `guidance` | number | `5.0` | Native model sampling guidance/CFG; omit for variant defaults. |
| `seed` | integer | `42` | Sampling noise seed; keep fixed to compare checkpoints. |
| `attention_backend` | "torch", "kitchen-int8" | `"torch"` | torch uses native attention; kitchen-int8 requires the relevant backend and supported inference paths. |
| `conditioning_image` | string / null | `null` | Single conditioning image path, e.g. FLUX Kontext/Fill or Wan I2V; follow family constraints. |
| `control_video` | string / null | `null` | Wan control-video path, only for applicable I2V control inputs. |

## `wan-generate.high_noise`

| Field | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `weights` | string | `required` | Local weight file, shard index or supported model directory; replace the example path. |
| `boundary` | number | `required` | Normalized Wan2.2 high/low-noise expert switch boundary, 0..1. |
| `guidance` | number / null | `null` | In sampling, model guidance/CFG; method.guidance is a training embedding for supported models. |
| `adapter` | object / null | `null` | Adapter table: training algorithm under train.adapter; inference weights and config under adapter. |

## `h3-generate.inputs`

| Field | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `prompt` | string | `required` | Prompt used for inference or the preview. |
| `first_frame` | string / null | `null` | First-frame condition path for H3. |
| `last_frame` | string / null | `null` | Last-frame condition path for H3. |
| `controls` | [object, …] | `[]` | H3 control-image records with time indices. |
| `target_index` | integer / null | `null` | Target frame index; match the record and conditioning indices. |
| `references` | [table, …] | `[]` | Ordered H3 image/video/audio reference records. |

## `h3-generate.sampling`

| Field | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `width` | integer | `768` | Output width in pixels; generally align to 16, or 32 for Qwen2.1/H3. |
| `height` | integer | `1344` | Output height in pixels; together with width determines sampling memory use. |
| `frames` | integer | `124` | Generated frame count; obey family constraints, with fps determining duration. |
| `fps` | integer | `24` | Generation frame rate; H3 uses native 24 FPS. |
| `temporal_fine_bands` | integer | `0` | Native H3 temporal frequency-band setting; normally retain the model default. |
| `allow_experimental_duration` | boolean | `false` | Allow experimental H3 durations; keep false for standard recipes. |
| `steps` | integer | `30` | Sampling iterations; omission uses variant defaults, do not reuse one value across Turbo and nondistilled models. |
| `seed` | integer | `42` | Sampling noise seed; keep fixed to compare checkpoints. |
| `attention_backend` | "torch", "kitchen-int8" | `"torch"` | torch uses native attention; kitchen-int8 requires the relevant backend and supported inference paths. |
| `video_shift` | number | `12.0` | H3 video schedule shift, default 12.0; match the training/sampling recipe. |
| `audio_shift` | number | `3.0` | H3 audio schedule shift, default 3.0. |
| `visual_condition_clean` | number | `0.999` | H3 visual-condition cleanliness; default 0.999. |
| `audio_condition_clean` | number | `1.0` | H3 audio-condition cleanliness; default 1.0. |

## `mid-timestep.dataset.defaults`

| Field | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `resolution` | [integer, integer] | `[1024, 1024]` | [width, height] in pixels; with buckets enabled this sets target area, not one fixed crop size. |
| `repeats` | integer | `1` | Repeat each item per data pass; increases a group's relative frequency, not max_updates. |
| `horizontal_flip_probability` | number | `0.5` | Paired horizontal-flip probability for restoration, 0..1; use 0 for direction-sensitive tasks. |

## `mid-timestep.dataset.groups.NAME`

| Field | Type / choices | Default | How to fill |
| --- | --- | --- | --- |
| `resolution` | [integer, integer] | `[1024, 1024]` | [width, height] in pixels; with buckets enabled this sets target area, not one fixed crop size. |
| `repeats` | integer | `1` | Repeat each item per data pass; increases a group's relative frequency, not max_updates. |
| `horizontal_flip_probability` | number | `0.5` | Paired horizontal-flip probability for restoration, 0..1; use 0 for direction-sensitive tasks. |
| `source` | string | `required` | Dataset directory or family-specific record file; rebuild caches after editing inputs. |
| `low_quality_source` | string / null | `null` | LQ restoration directory, stem-matched to HQ source; mutually exclusive with synthetic degradation. |
| `degradation` | object / null | `null` | Two-stage synthetic LQ degradation; mutually exclusive with this group's low_quality_source. |

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
