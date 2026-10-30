# Wan and MiniMax-H3 training tutorial

[简体中文](../../training/video.md) · [Parameters](parameters.md) · [Model assets](models.md)

## Wan: cache before training

Copy `docs/examples/wan-cache.toml` and `docs/examples/wan-flow.toml`. Fill encoder assets and the video directory in the cache file, then denoiser weights, matching data/cache paths and optimization settings in the training file. Use wan-flow-preview.toml for training previews and retain its VAE/text encoder paths.

| Location | What to fill |
| --- | --- |
| model.version | 2.1 or 2.2, matching the checkpoint |
| model.variant | t2v or i2v, with appropriate I2V conditions |
| model.vae / text_encoder | Local Wan VAE and UMT5 weights |
| model.assets | Wan2.1 I2V auxiliary root containing image_encoder/model.safetensors |
| dataset.groups.main | Video source, dedicated cache_directory, resolution and batch_size |
| dataset.groups.main.video | target_frames, extraction, target_fps and fps_resampling |

```bash
uv run --no-dev unituner cache wan-cache.toml --device cuda
uv run --no-dev unituner validate wan-train.toml
uv run --no-dev unituner train wan-train.toml --device cuda
```

Name your copies wan-cache.toml and wan-train.toml. Wan target_frames and max_frames must follow 1+4k, such as 81 or 129. Frame count and resolution jointly affect memory: start with short clips rather than extrapolating from image batch sizes.

head extracts a window from the start; alternatives are chunk, slide, uniform and full. Combine stride/samples for multiple windows; see [defaults](parameters.md). Copy wan-single-frame-cache.toml for image training with independent condition frames instead of mixing that setup with normal video controls.

## MiniMax-H3: AV assets and records

Copy minimax-h3-cache.toml and minimax-h3-train.toml. H3 consumes audio/video records; an arbitrary video directory is not a native AV JSONL input.

| Location | What to fill |
| --- | --- |
| model.variant | t2va, fl2va or ref2va, matching conditions |
| model.vae | Local video VAE |
| model.minimax_h3.audio_vae | Separate audio VAE |
| model.text_encoder / minimax_h3.text_config | Matching text weights and topology configuration |
| model.minimax_h3.text_processor | Local processor when available; family defaults are also supported |
| dataset.groups.main.source | Native records.jsonl referencing actual targets/conditions |
| dataset.groups.main.video | target_fps=24.0, fps_resampling="timestamps", target frames following 17n+5 |

H3 uses batch_size=1, 32-aligned dimensions and native 32 kHz stereo audio. The example's 124 frames satisfy 17n+5. Keep the standard duration policy unless intentionally testing allow_experimental_duration.

```bash
uv run --no-dev unituner cache h3-cache.toml --device cuda
uv run --no-dev unituner validate h3-train.toml
uv run --no-dev unituner train h3-train.toml --device cuda
```

Cache/train configurations must reference the same data and cache_directory. method.minimax_h3 controls video/audio schedule shifts, condition cleanliness and audio loss weight. Start with the example recipe and vary one setting at a time.

## Previews, saving and inference

Wan uses preview.videos array tables; H3 uses inputs/sampling within preview.audio_video. Copy the respective train-preview examples and fill model-specific conditions and sampling fields. Do not add image preview.samples.

Save cadence, separate inference weights/resume files, retention and TensorBoard work as in image training. Resume uses resume/*.state-*.safetensors; generation loads inference adapters/bases from weights/.

```bash
uv run --no-dev unituner generate-video wan-generate.toml --device cuda
uv run --no-dev unituner generate-video h3-generate.toml --device cuda
```

Copy wan-generate-video.toml and minimax-h3-generate.toml for inference. Fill output paths, matching bases/adapters, native frame counts and conditions. Add Wan2.2 high_noise settings only when supplying a second expert's weights.

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
