# Wan 与 MiniMax-H3 训练教程

[English](../en/training/video.md) · [训练参数](parameters.md) · [模型资产](models.md)

## Wan：先缓存，再训练

复制 `docs/examples/wan-cache.toml` 和 `docs/examples/wan-flow.toml`。前者填写编码资产和视频目录，后者填写 denoiser、同一组数据/缓存路径和优化设置。需要训练中预览时改用 `wan-flow-preview.toml`，同时保留 VAE 和文本编码器路径。

| 位置 | 填写要求 |
| --- | --- |
| model.version | 2.1 或 2.2，与实际 checkpoint 匹配 |
| model.variant | t2v 或 i2v；I2V 需要对应条件 |
| model.vae / text_encoder | Wan VAE 与 UMT5 本地权重 |
| model.assets | Wan2.1 I2V 的辅助根目录，包含 image_encoder/model.safetensors |
| dataset.groups.main | 视频 source、独立 cache_directory、resolution、batch_size |
| dataset.groups.main.video | target_frames、extraction、target_fps、fps_resampling |

```bash
uv run --no-dev unitunner cache wan-cache.toml --device cuda
uv run --no-dev unitunner validate wan-train.toml
uv run --no-dev unitunner train wan-train.toml --device cuda
```

将复制出的文件分别命名为 `wan-cache.toml` 和 `wan-train.toml`。Wan 的 target_frames 和 max_frames 必须满足 1+4k，例如 81、129。帧数和分辨率共同影响显存；先用短片段验证，不要仅根据图像训练的 batch 推断视频需求。

提取策略 head 从开头取窗口；其他可选值为 chunk、slide、uniform、full。需要多个窗口时结合 stride/samples，具体默认值见[参数表](parameters.md)。独立条件帧的图片训练复制 `wan-single-frame-cache.toml`，不要混入普通视频控制目录。

## MiniMax-H3：准备 AV 资产和记录

复制 `minimax-h3-cache.toml` 与 `minimax-h3-train.toml`。H3 使用音视频记录，不能将任意视频目录直接当成原生 AV JSONL。

| 位置 | 填写要求 |
| --- | --- |
| model.variant | t2va、fl2va、ref2va，匹配任务条件 |
| model.vae | 本地视频 VAE |
| model.minimax_h3.audio_vae | 独立音频 VAE |
| model.text_encoder / minimax_h3.text_config | 匹配的文本权重和结构配置 |
| model.minimax_h3.text_processor | 已有本地 processor 时填写；也可使用模型默认来源 |
| dataset.groups.main.source | 原生 records.jsonl；记录指向真实目标与参考文件 |
| dataset.groups.main.video | target_fps=24.0、fps_resampling="timestamps"、17n+5 目标帧数 |

H3 使用 batch_size=1、32 对齐分辨率和原生 32 kHz 双声道音频。示例 124 帧满足 17n+5。保留标准时长策略；只有主动测试实验时长时才启用 allow_experimental_duration。

```bash
uv run --no-dev unitunner cache h3-cache.toml --device cuda
uv run --no-dev unitunner validate h3-train.toml
uv run --no-dev unitunner train h3-train.toml --device cuda
```

缓存与训练配置指向同一组数据和 cache_directory。method.minimax_h3 控制视频/音频时间移位、条件清洁度和音频损失权重；先保持示例配方，再单独调整。

## 预览、保存与推理

Wan 使用 `[[preview.videos]]`，H3 使用 `[[preview.audio_video]]` 下的 inputs/sampling。复制各自 train-preview 示例，按同一模型填写条件和采样设置；不要追加图像的 preview.samples。

保存周期、正式权重/续训状态分离、保留数量和 TensorBoard 与图像训练相同。续训加载 resume/*.state-*.safetensors；生成加载 weights/ 下的正式 LoRA 或基模。

```bash
uv run --no-dev unitunner generate-video wan-generate.toml --device cuda
uv run --no-dev unitunner generate-video h3-generate.toml --device cuda
```

推理配置分别复制 `wan-generate-video.toml` 和 `minimax-h3-generate.toml`。填写输出文件、同一底模、适配器、原生采样帧数和条件。Wan2.2 双专家生成的 high_noise 设置只在需要第二个专家权重时添加。

---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
