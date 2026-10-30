# 推理与模型使用

简体中文 · [English](en/inference.md) · [文档首页](README.md)

训练完成后，先确定手里是适配器、完整模型，还是训练状态。它们决定如何加载，而生成、视频和修复任务则分别使用自己的推理配置。

<details>
<summary>本页导航</summary>

- [选择正确的文件](#选择正确的文件)
- [用图像 LoRA 生成](#用图像-lora-生成)
- [使用 full 模型](#使用-full-模型)
- [一步图像修复](#一步图像修复)
- [视频与音视频](#视频与音视频)
- [在 ComfyUI 中加载](#在-comfyui-中加载)
- [常见检查](#常见检查)

</details>

## 选择正确的文件

| 文件 | 用途 | 加载方式 |
| --- | --- | --- |
| `<name>.safetensors`，adapter 训练 | 模型权重增量 | 保留原底模，用 `[adapter]` 加载 |
| `<name>.safetensors`，full 训练 | 训练后的完整模型 | 替换 `model.weights`，不配置 adapter |
| `<name>.encoder.safetensors` | 修复编码器增量 | 在 `restore` 的 `encoder_weights` 中指定 |
| `<name>.state-*.safetensors` | 优化器与完整训练状态 | 仅供 `train --resume`，不用于生成 |

仅凭扩展名无法判断用途，要结合训练方式和文件名。适配器需要与训练时的基模、算法、rank 和 alpha 匹配。

若训练使用量化基模，推理应加载同一量化文件。训练前自动量化的运行需要先用相同格式、旋转／分组参数和设备设置[转换基模](weight-conversion.md)，再把推理配置的 `model.weights` 指向该文件；推理配置没有训练用的 `execution.quantization`。预览直接使用训练中的模型，独立生成改回原浮点基模可能改变输出。

文本编码器也可加载提前量化的文件，或通过 `[model.text]` 在推理前逐层量化，并独立选择 FP32／FP16／BF16 浮点计算。保留对应 tokenizer、processor 和源键格式；详见[文本编码](training/text-encoding.md)。

## 用图像 LoRA 生成

[第一次训练](getting-started.md)的 `garden` LoRA 对应以下完整推理配置，保存为 `generate.toml`：

```toml
output = "runs/garden/sample.png"

[model]
family = "z-image"
variant = "base"
weights = "models/z-image.safetensors"
vae = "models/z-image-vae.safetensors"
text_encoder = "models/qwen3"

[adapter]
weights = "runs/garden/weights/garden.safetensors"

[adapter.config]
algorithm = "lora"
rank = 32
alpha = 32.0

[sampling]
prompt = "a garden path in the style of gardenstyle, soft morning light"
negative_prompt = ""
width = 1024
height = 1024
steps = 20
guidance = 4.0
seed = 42
```

也可直接复制[配置文件](examples/tutorial-generate.toml)，然后运行：

```bash
uv run unituner generate generate.toml --device cuda
```

`seed` 固定初始随机噪声，便于在相同配置下比较模型；`steps` 控制采样迭代次数；`width`、`height` 控制输出尺寸。`guidance` 的语义取决于模型：有的采用 CFG，有的采用 learned guidance embedding，蒸馏变体还可能固定或不接受该字段。更换模型时应从对应[生成示例](examples/README.md#推理)开始。

为了比较底模与 LoRA，可以复制配置，删除 `[adapter]` 与 `[adapter.config]`，并使用不同输出文件。其他设置保持一致，便于观察微调带来的变化。

## 使用 full 模型

复制对应家族的生成配置，把 `model.weights` 改为训练后的完整模型路径。保留对应 VAE、文本编码器与条件资产，删除适配器段。生成命令不变。

Full 训练即使只选择部分 denoiser 层，仍发布完整模型。不要再将同一轮 full 产物当成 LoRA 叠加到原底模上。

## 一步图像修复

修复使用 `restore`，完整流程见[修复教程](restoration-tutorial.md)。配置中指定输入图片、输出路径，以及对应的 denoiser／encoder 产物：

```bash
uv run unituner restore restore.toml --device cuda
```

默认从训练产物读取 sigma。Denoiser 和 encoder 需要来自匹配的训练结果，避免混用不同恢复时间。修复推理使用后验均值；启用 encoder 训练的结果需配套加载 encoder 文件。

## 视频与音视频

Wan 和 MiniMaxH3 使用 `generate-video`：

```bash
uv run unituner generate-video docs/examples/wan-generate-video.toml --device cuda
uv run unituner generate-video docs/examples/minimax-h3-generate.toml --device cuda
```

运行前替换示例中的本地资产与输出路径。分辨率、帧数、音频和参考条件遵循各家族的约束，不能直接复用图像配置中的 sampling 段。见[模型指南](training/models.md#视频与音视频)。

## 在 ComfyUI 中加载

当前 adapter 训练默认发布对应模型的 ComfyUI 权重键，可作为该基模的 LoRA／LoHa／LoKr 使用。ComfyUI 本身需要支持相应模型以及融合投影映射。修复 encoder 文件属于 UniTuner 的修复流程，不能作为普通 denoiser LoRA 加载。

`adapters export-comfyui` 用于转换已有 Wan／H3 原生格式适配器；当前训练发布的适配器通常不需要再转换。具体格式说明见[训练指南](training/README.md#推理与发布)。

## 常见检查

生成与修复配置不交给 `unituner validate`；该命令只验证训练配置。对应推理命令会加载自己的配置。若无法运行，依次检查模型家族与变体、底模路径、适配器算法和 rank、条件资产，再检查尺寸及采样参数。

推理默认使用 Torch 注意力。可选 Kitchen INT8 会改变数值输出，若需测试，见[INT8 推理注意力](training/execution.md#推理时使用-int8-注意力)。

---

[文档首页](README.md) · [参数参考](training/parameters.md) · [故障排查](troubleshooting.md)
