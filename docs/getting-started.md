# 从图片到 LoRA

[English](en/getting-started.md) · [文档首页](README.md)

本教程使用 Z-Image Base 完成图片准备、配置、缓存、训练、预览和推理。使用其他模型时，复制[对应示例](examples/README.md)，同时替换模型、VAE、文本编码器和变体。

[环境安装详解](installation.md) · [TOML 与 CLI](configuration.md) · [故障排查](troubleshooting.md)

<details>
<summary>本页导航</summary>

- [1. 安装](#1-安装)
- [2. 填写模型路径](#2-填写模型路径)
- [3. 准备图片和提示词](#3-准备图片和提示词)
- [4. 配置训练长度、容量和精度](#4-配置训练长度容量和精度)
- [5. 配置观察与保存](#5-配置观察与保存)
- [6. 校验、缓存和训练](#6-校验缓存和训练)
- [7. 查看结果和恢复](#7-查看结果和恢复)
- [8. 用 LoRA 生成测试图](#8-用-lora-生成测试图)

</details>

## 1. 安装

在已配置兼容 CUDA 驱动的环境中运行：

```bash
git clone https://github.com/nukuihayu/UniTunner.git
cd UniTunner
uv python install 3.13
uv sync --locked --no-dev --extra media --extra text
uv run --no-dev unitunner --help
```

`media` 提供图像/视频/音频读写，`text` 提供文本编码与分词依赖。后续命令都在仓库根目录执行。

国内环境需要清华 PyPI 源时，安装依赖使用：

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv sync --locked --no-dev --extra media --extra text
```

该地址用于 Python 包，不是模型下载源或 Python 解释器下载源。依赖安装完后用 `uv run --no-sync --no-dev` 启动可避免训练前再次同步依赖。

## 2. 填写模型路径

准备 denoiser、匹配的 VAE、文本编码器权重，然后复制示例：

```bash
cp docs/examples/tutorial-lora.toml train.toml
```

修改已有 `[model]` 表：

```toml
[model]
family = "z-image"
variant = "base"
weights = "/models/z-image/diffusion_models/z_image_bf16.safetensors"
vae = "/models/z-image/vae/ae.safetensors"
text_encoder = "/models/z-image/text_encoders/qwen_3_4b.safetensors"
```

对应家族支持的 Diffusers／ComfyUI 权重可以直接填写，不需先转换为训练器专用格式。`variant` 必须匹配权重；Turbo 权重改用 `turbo`，并同步调整预览采样设置。

分词器与文本编码器权重分开。默认使用模型内置来源获取分词文件；已有本地分词器时添加：

```toml
[model.text.tokenizer]
source = "/models/z-image/tokenizer"
offline = true
```

路径应直接包含 tokenizer 文件。若填写仓库根目录而文件在其子目录中，再填写 `subfolder = "tokenizer"`。本地编码器权重不会随分词器自动下载。

## 3. 准备图片和提示词

```text
data/garden/
├── 001.jpg
├── 001.txt
├── 002.png
└── 002.txt
```

图片放在源目录直属层级；同名 UTF-8 `.txt` 描述各自内容，例如 `a garden path, stone steps, flowers, gardenstyle`。未提供侧文件时使用空文本。

修改数据表：

```toml
[dataset.groups.main]
kind = "image"
source = "data/garden"
cache_directory = "cache/garden"
resolution = [1024, 1024]
batch_size = 1
repeats = 1
enable_bucket = true
```

`resolution` 为 `[宽, 高]`。开启分桶后按宽高比选择接近目标面积的尺寸；不是把所有图都强行拉伸到正方形。每个组必须有独立缓存目录。

所有图使用相同提示词时，在该表填 `prompt = "a garden in gardenstyle"`，不必复制大量同名 txt。它会覆盖已有 txt；填 `prompt = ""` 是明确使用空提示词。需要逐图提示词时删除 `prompt`。

多组数据和配对参考图见[数据集教程](training/datasets.md)。

## 4. 配置训练长度、容量和精度

合并到已有表：

```toml
[train]
parameter_mode = "adapter"
learning_rate = 1e-4
max_updates = 100
accumulation_steps = 1
seed = 42

[train.adapter]
algorithm = "lora"
rank = 32
alpha = 32.0

[execution]
precision = "bf16"
recompute = "block"
```

这是 CUDA BF16 起步配方。CPU 检查流程改用 `fp32`；`auto` 按底模精度解析，不是按设备自动降级。显存充足时比较 `recompute = "none"` 的稳定速度。

| 你要改变什么 | 修改字段 | 如何判断 |
| --- | --- | --- |
| 训练更久 | `max_updates` | 用固定提示词预览判断何时过拟合，先做短跑 |
| 增大适配器容量 | `rank`、`alpha` | 通常先保持 alpha=rank；同时增大后重新比较 |
| 减少每次前向显存 | `batch_size` | 先减到 1，再使用梯度累积 |
| 增加有效批量 | `accumulation_steps` | 单组约为 batch_size × accumulation_steps；一次更新会更久 |
| 降低更新幅度 | `learning_rate` | 比较细节、颜色和泛化，不只看总 loss |

## 5. 配置观察与保存

```toml
[output]
directory = "runs/garden"
name = "garden"
save_every_updates = 50
tensorboard = "runs/tensorboard"

[preview]
every_updates = 50
at_start = true

[[preview.samples]]
prompt = "a garden path in gardenstyle, soft morning light"
width = 1024
height = 1024
steps = 20
guidance = 4.0
seed = 42
```

`steps` 和 `guidance` 属于这个 Base 预览配方；不要直接套到 Turbo。预览需要 VAE 和文本编码器。固定提示词与种子便于比较不同保存步。

省略 `output.directory` 会自动创建带名称和时间戳的目录；显式目录会复用。全新训练会清理该目录中同名的旧训练产物，续训使用 `--resume`。

默认只保留最近一份 resume，保留全部正式权重，无需额外填写。需要保留多份续训状态时设置 `max_resume_checkpoints`；限制正式权重组数则设置 `max_weight_checkpoints`，两者 `0` 均表示不限。

## 6. 校验、缓存和训练

```bash
uv run --no-dev unitunner validate train.toml
uv run --no-dev unitunner cache train.toml --device cuda
uv run --no-dev unitunner train train.toml --device cuda
```

`validate` 检查字段和组合，不加载权重。`cache` 加载 VAE 与文本编码器，生成 latent/文本特征；训练读取缓存。修改图片、提示词、VAE、文本编码器或预处理后重建缓存。

每次执行 `cache` 都重新编码，成功后替换旧缓存；失败保留旧缓存。重建期间不要启动读取同一缓存的训练。

终端原位显示进度和指标。后台或重定向默认每 60 秒输出摘要，首步和末步必报；通常无需设置 `log_every_seconds`。完整分损失、梯度范数、学习率、耗时和显存写入 JSONL 与已启用的 TensorBoard。

## 7. 查看结果和恢复

100 步示例结束后：

```text
runs/garden/
├── weights/
│   ├── garden-00000050.safetensors
│   ├── garden-00000100.safetensors
│   └── garden.safetensors
├── resume/
│   └── garden.state-00000100.safetensors
├── previews/
└── logs/garden.jsonl
```

`garden.safetensors` 跟随每次导出更新，编号文件用于保留训练版本。只有 `resume/` 中的状态用于精确续训；普通 LoRA 文件不含优化器和随机状态。

```bash
uv run --no-dev unitunner train train.toml --device cuda --resume runs/garden/resume/garden.state-00000100.safetensors
tensorboard --logdir runs/tensorboard --host 0.0.0.0 --port 6006
```

续训继续到配置中的 `max_updates` 总数，不是额外再训练该数量。恢复已完成的最后一步不会产生新的更新。模型、数据和影响数值的训练设置必须通过续训身份检查。

## 8. 用 LoRA 生成测试图

```bash
cp docs/examples/tutorial-generate.toml generate.toml
uv run --no-dev unitunner generate generate.toml --device cuda
```

在推理配置中填写相同底模/VAE/文本编码器；`adapter.weights` 填 `runs/garden/weights/garden.safetensors`，算法、rank、alpha 与训练一致。修改 `[sampling]` 中的提示词、尺寸和种子。检查输出路径，不要覆盖输入图片。

下一步：[完整训练参数](training/parameters.md) · [优化显存与速度](training/execution.md) · [图像修复](restoration-tutorial.md)。

---

[文档首页](README.md) · [参数参考](training/parameters.md) · [故障排查](troubleshooting.md)
