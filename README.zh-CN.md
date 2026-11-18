<p align="center">
  <img src="docs/assets/unituner-logo.svg" alt="UniTuner" width="720">
</p>
<p align="center"><strong>图像、视频与音视频模型训练工具</strong></p>
<p align="center"><a href="README.md">English</a> · 简体中文</p>
<p align="center">
  <a href="docs/getting-started.md">快速开始</a> ·
  <a href="docs/README.md">使用文档</a> ·
  <a href="docs/training/models.md">支持模型</a> ·
  <a href="docs/training/parameters.md">参数参考</a> ·
  <a href="docs/examples/README.md">配置示例</a>
</p>

---

UniTuner 将数据编码、模型微调、训练预览和权重导出连接为一个可配置的工作流。使用 **LoRA、LoHa、LoKr 或全参数训练**学习主体、风格和条件任务；也可以使用配对图像或合成退化，训练一步图像修复模型。

用 TOML 保存训练配置，用命令行覆盖单次运行参数。选择对应模型的示例，填写本地模型与数据路径即可开始。

**阅读导航：** [快速开始](#快速开始) · [支持模型](#支持的模型) · [训练步骤](#从数据到模型) · [显存设置](#显存精度与执行策略)

## 快速开始

需要 Python **3.12+**、[uv](https://docs.astral.sh/uv/) 和本地模型资产。CUDA 训练还需要兼容项目 Torch 版本的 NVIDIA 驱动；[安装指南](docs/installation.md)说明默认依赖与环境检查。

```bash
git clone https://github.com/nukuihayu/UniTuner.git
cd UniTuner
uv python install 3.13
uv sync --locked --no-dev
cp docs/examples/tutorial-lora.toml train.toml
```

编辑 `train.toml` 中的模型、VAE、文本编码器和数据路径，再执行：

```bash
uv run --no-dev unituner validate train.toml
uv run --no-dev unituner cache train.toml --device cuda
uv run --no-dev unituner train train.toml --device cuda
```

> [!TIP]
> 这是 Z-Image Base LoRA 的入门流程。第一次使用请跟随[从图片到 LoRA](docs/getting-started.md)，包含数据目录、参数解释、预览和生成测试图。其他模型使用[对应示例](docs/examples/README.md)。

### 配置保持简单

在已有配置表中修改字段，CLI 参数覆盖 TOML：

```bash
uv run --no-dev unituner train train.toml --device cuda \
  --train.learning-rate 5e-5 --train.max-updates 2000
```

常用参数按职责分组：

| 配置表 | 你要决定的内容 |
| --- | --- |
| `[model]` | 模型家族、变体、本地权重与编码资产 |
| `[dataset.groups.NAME]` | 数据来源、缓存、尺寸、batch 和 repeats |
| `[train]`、`[train.adapter]` | 更新范围、算法、rank、学习率与训练长度 |
| `[method]` | 流匹配或修复，以及可选辅助目标 |
| `[execution]` | 计算精度、重算、权重驻留和编译 |
| `[preview]`、`[output]` | 观察条件、保存周期、日志和保留数量 |

详细规则见 [TOML 与 CLI](docs/configuration.md)；查默认值和范围使用[参数参考](docs/training/parameters.md)。

## 能做什么

| 能力 | 使用方式 |
| --- | --- |
| 生成模型微调 | 图像、视频、音视频流匹配；支持适配器与全参数 |
| 一步图像修复 | LQ/HQ 配对或合成退化，时间校准、VAE 编码器适配与多种损失 |
| 多数据集 | 独立数据组、共享默认值、提示词、重复次数、尺寸与长宽比分桶 |
| 显存管理 | 块重算、注意力激活卸载、冻结权重卸载、冻结底模量化 |
| 观察与续训 | 固定条件预览、JSONL、TensorBoard、独立续训状态与保留策略 |
| 模型使用 | 图像/视频生成、修复推理、适配器导出与权重精度转换 |

## 支持的模型

| 任务类型 | 家族 |
| --- | --- |
| 图像 | Z-Image · FLUX · FLUX.2 · FLUX3 · MOSS Image · Qwen Image · Qwen Image 2.1 · Krea2 · MiniMax-H3-Image |
| 视频 | Wan 2.1 / 2.2 |
| 音视频 | MiniMaxH3 |

各模型的图像修复支持范围、编辑/图层条件、变体和所需文件见[模型与变体](docs/training/models.md)。MiniMax-H3-Image 的缓存、训练和推理见[专用指南](docs/minimax-h3-image.md)。选择配置时必须同时匹配家族、变体和权重，不能只替换模型名称。

## 从数据到模型

```text
模型资产 + 图片/视频/音视频 + 文本与条件
                    │
                 validate    检查配置与组合
                    │
                  cache      编码 latent、文本和条件
                    │
                  train      训练模型并保存结果
                    ├── previews/   固定条件观察效果
                    ├── logs/       损失、学习率和资源指标
                    ├── resume/     继续同一次训练
                    └── weights/    生成、修复或外部加载
```

数据集使用具名分组，每组可以独立指定来源、缓存目录、分辨率、batch 和 repeats，并通过 `dataset.defaults` 共享设置。图像支持同名 caption、统一 prompt、长宽比分桶，以及模型要求的参考图、掩码或图层。视频和音视频使用各自的帧数、帧率、音频与条件规则。

编码缓存让普通流匹配训练避免在每次更新时重复运行 VAE 和文本编码器。更改图片、提示词、编码器或预处理设置后需要重新编码。修复训练仍实时处理图像和 VAE，具体数据准备见[数据集指南](docs/training/datasets.md)。

## 显存、精度与执行策略

模型大小、训练尺寸、可训练参数和损失网络共同决定资源需求。UniTuner 将不同资源的管理方式分别配置：

| 策略 | 解决的问题 | 使用条件 |
| --- | --- | --- |
| Resident | 权重始终留在计算设备 | 默认策略，适用于 adapter 与 full |
| 块重算 | 减少长期保存的前向激活 | 减少显存占用，增加计算耗时 |
| 注意力激活卸载 | 将注意力保存的激活暂存到主机 | CUDA；增加 RAM 和传输需求 |
| H2D 冻结块缓存 | 按需载入冻结权重，并复用允许常驻的块 | CUDA adapter；模式内自动校准缓存容量 |
| 冻结底模量化 | 降低冻结权重存储需求 | 支持的量化布局与 adapter 训练 |
| CUDA 编译 | 尝试提高训练速度 | 可选启用，首次调用有编译开销 |

计算精度可使用 FP32、CUDA BF16 或 FP16；默认 `auto` 根据底模文件解析。可训练主参数保持 FP32，FP16 使用固定 loss scaling。文本编码精度和导出精度独立设置。驻留模式需要显式选择，`resident` 不会在显存不足时自动切换到卸载。

默认 eager 与 resident 可以作为起步配置，再按实际需要选择[执行策略](docs/training/execution.md)、[精度](docs/training/precision.md)和[量化底模](docs/training/quantized-base.md)。

## 训练产物

```text
runs/<运行名>/
├── weights/     # 推理模型、适配器及编号导出
├── resume/      # 模型/方法、优化器、随机状态与数据游标
├── previews/    # 训练过程的生成或修复结果
└── logs/        # 完整日志与训练指标
```

使用 `weights/` 中的文件进行推理；使用 `resume/` 中的状态继续训练：

```bash
uv run --no-dev unituner train train.toml --device cuda \
  --resume runs/garden/resume/garden.state-00000050.safetensors
```

> [!IMPORTANT]
> `max_updates` 是目标总更新数。普通适配器文件不包含完整续训状态。每次 `cache` 会重新编码并在成功后替换旧缓存；不要同时训练读取该缓存。

## 观察、发布与使用结果

训练预览按配置周期生成，使用固定提示词、种子和条件。JSONL 保存每次更新的指标；可选 TensorBoard 展示损失、梯度范数、学习率和显存使用。保存与预览周期独立设置，可以调整观察频率而不改变训练目标。

在 ComfyUI 中使用训练结果时，选择匹配的底模和支持对应适配器算法的加载器。也可以使用 UniTuner 的 `generate`、`generate-video` 或 `restore`。跨工具比较效果时，应对齐底模、VAE、文本资产、条件输入与采样约定。

权重工具还支持导入适用的外部布局、转换浮点精度及处理支持的量化格式，并按模型保护敏感层。格式与命令示例见[推理指南](docs/inference.md)和[权重转换](docs/weight-conversion.md)。

## 按任务阅读

| 我想…… | 指南 |
| --- | --- |
| 完成第一次训练 | [安装](docs/installation.md) → [LoRA 教程](docs/getting-started.md) |
| 调整训练和数据 | [训练指南](docs/training/README.md) · [数据集](docs/training/datasets.md) |
| 训练图像修复 | [修复教程](docs/restoration-tutorial.md) · [损失与时间参数](docs/training/restoration.md) |
| 训练视频或音视频 | [Wan / MiniMaxH3](docs/training/video.md) |
| 解决显存不足 | [执行策略](docs/training/execution.md) · [量化底模](docs/training/quantized-base.md) |
| 用训练好的模型 | [推理与 ComfyUI](docs/inference.md) · [权重转换](docs/weight-conversion.md) |
| 理解报错或异常结果 | [故障排查](docs/troubleshooting.md) · [日志](docs/logging.md) |

文档提供[中文](docs/README.md)和[英文](docs/en/README.md)两个版本，共用可执行 TOML 示例。
