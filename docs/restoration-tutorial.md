# 用现成图片训练一步修复 LoRA

[English](en/restoration-tutorial.md) · [训练教程](training/README.md)

本教程直接使用真实 LQ/HQ 图片，不构造退化。先用 LRR＋L1 验证流程，再按需要加入 DINOv3、GAN 和 VGG 损失。

<details>
<summary>本页导航</summary>

- [1. 准备配对数据](#1-准备配对数据)
- [2. 填写数据和训练参数](#2-填写数据和训练参数)
- [3. 固定时间或自动校准，选择一种](#3-固定时间或自动校准选择一种)
- [4. 预览、缓存与训练](#4-预览缓存与训练)
- [5. 添加感知、颜色或对抗损失](#5-添加感知颜色或对抗损失)
- [6. 修复一张新图片](#6-修复一张新图片)
- [7. 续训](#7-续训)

</details>

## 1. 准备配对数据

```text
data/restoration/
├── hq/a.png
├── hq/b.png
├── lq/a.png
├── lq/b.png
└── test.png
```

HQ 是希望恢复成的目标，LQ 是实际输入。按文件主体配对，检查尺寸、构图和位置已对齐。test.png 留作推理验证，避免只检查训练图。

```bash
cp docs/examples/tutorial-restoration.toml restoration.toml
```

修改 model.weights、vae、text_encoder 以及 tokenizer 来源；[模型路径教程](getting-started.md#2-填写模型路径)说明文件形式。

## 2. 填写数据和训练参数

```toml
[dataset.groups.main]
kind = "restoration"
source = "data/restoration/hq"
low_quality_source = "data/restoration/lq"
cache_directory = "cache/restoration"
resolution = [1024, 1024]
batch_size = 1
prompt = ""
horizontal_flip_probability = 0.0

[train]
parameter_mode = "adapter"
learning_rate = 5e-5
max_updates = 100
accumulation_steps = 4
seed = 42

[train.adapter]
algorithm = "lora"
rank = 32
alpha = 32.0

[method]
kind = "restoration"

[method.restoration]
midtimestep = 400
calibrate = false
encoder_rank = 4

[losses]
lrr = 3.0
l1 = 0.5
dv3d = 0.0
gan = 0.0
fdl = 0.0
histogram = 0.0

[execution]
precision = "bf16"
recompute = "block"
```

这是固定时间 CUDA BF16 起步配方，400 只是可比较的实验值，不是所有数据的最佳时间。prompt="" 让所有图片使用空文本；需要统一描述时直接替换字符串，不用创建 txt。不要添加 degradation 表。

encoder_rank=4 训练独立的 VAE encoder adapter；0 禁用。它与 denoiser LoRA 分开导出，推理时要成套使用。

## 3. 固定时间或自动校准，选择一种

| 策略 | 填写方式 | 适合场景 |
| --- | --- | --- |
| 固定整数时间 | midtimestep=400，calibrate=false，不填 sigma | 对照时间点实验 |
| 固定 sigma | sigma=某个 (0,1) 数值，calibrate=false | 已得到校准结果，固定噪声水平 |
| 训练前自动校准 | calibrate=true，可填 calibration_pairs=32，不填 sigma | 从当前 LQ/HQ 估计噪声水平 |

整数 midtimestep 经模型时间表变换为 sigma；不能对所有家族直接除以 1000。自动校准时使用估计值，而不是同时使用固定整数时间作为第二个训练时间。

仅检查校准，不开始训练：

```bash
uv run --no-dev unitunner mid-timestep restoration.toml --device cuda
```

命令只读取 VAE 和配对图，不加载 denoiser/文本编码器，不需要缓存，输出 JSON，不自动改 TOML。需要保留结果时手动把输出 sigma 合并进 method.restoration，并关闭 calibrate。修改数据、尺寸或 VAE 后重新评估校准。

## 4. 预览、缓存与训练

```toml
[output]
directory = "runs/restoration"
name = "restoration"
save_every_updates = 50
tensorboard = "runs/tensorboard"

[preview]
every_updates = 25
at_start = true
num_images = 1
```

修复预览不填 preview.samples。它轮换真实数据，输出 LQ/恢复结果/HQ 对比。固定 seed 保持选图顺序，重点看输入一致性、肤色、纹理和过度锐化。

```bash
uv run --no-dev unitunner validate restoration.toml
uv run --no-dev unitunner cache restoration.toml --device cuda
uv run --no-dev unitunner train restoration.toml --device cuda
```

修复训练实时读取原始 LQ/HQ 并运行可训练 encoder 和图像损失，缓存后不要删除源图。

100 步结束后的主要文件：

```text
runs/restoration/
├── weights/
│   ├── restoration-00000050.safetensors
│   ├── restoration-00000050.encoder.safetensors
│   ├── restoration-00000100.safetensors
│   ├── restoration-00000100.encoder.safetensors
│   ├── restoration.safetensors
│   └── restoration.encoder.safetensors
├── resume/restoration.state-00000100.safetensors
├── previews/
└── logs/restoration.jsonl
```

正式权重默认保留每次导出，resume 默认只保留最近一个。修改保留上限见[保存与恢复](training/README.md#保存与恢复)。

## 5. 添加感知、颜色或对抗损失

确认基础流程后，按需修改，不必全部开启：

| 字段 | 作用 | 额外填写 |
| --- | --- | --- |
| losses.lrr | latent 恢复目标 | 无外部感知权重 |
| losses.l1 | 像素一致性 | 无 |
| losses.dv3d | DINOv3 特征距离 | method.restoration.dino_weights |
| losses.gan | 对抗细节学习，增加判别器更新 | 同一 DINOv3 ConvNeXt 权重 |
| losses.fdl | VGG 频率分布目标 | method.restoration.vgg_weights |
| losses.histogram | 颜色分布 | 可调 histogram_bins，无外部权重 |

DINO 这里使用 ConvNeXt，不是任意 DINO ViT。默认 dino_widths/depths 匹配 ConvNeXt Large；VGG 默认通道匹配 VGG19。权重结构要对应，不能仅改文件名。

完整感知配方见 [restoration.toml](examples/restoration.toml)，油光肤色 Z-Image Turbo 配方见 [oil-skin-lora.toml](examples/oil-skin-lora.toml)。后者含真实路径，复制后按自己的目录修改。

TensorBoard 逐步记录生成器、判别器及各分损失。总损失下降不代表每一项都改善；更换损失系数后，数值尺度也变，不能跨配方只比较总 loss。小数据出现纹理伪影时先比较更早的导出版本。

## 6. 修复一张新图片

```bash
cp docs/examples/tutorial-restore.toml restore.toml
uv run --no-dev unitunner restore restore.toml --device cuda
```

修改根级 input/output/prompt 和模型路径；adapter.weights 填训练 LoRA，encoder_weights 填同一步 encoder adapter。固定使用编号文件可避免训练继续保存时引用到不同版本。

```toml
input = "data/restoration/test.png"
output = "runs/restoration/result.png"
prompt = ""
encoder_weights = "runs/restoration/weights/restoration-00000100.encoder.safetensors"
```

训练产物已记录 sigma，推理通常不需要再填。恢复过程使用后验均值，保留输入尺寸。若 encoder_rank=0，删除 encoder_weights；full 训练则把 model.weights 改为导出基模，并删除 adapter 表。

## 7. 续训

```bash
uv run --no-dev unitunner train restoration.toml --device cuda --resume runs/restoration/resume/restoration.state-00000050.safetensors
```

此命令适用于在第 50 步后中断、该文件仍为最新 resume 的情况；完成 100 步后默认只剩 100 步状态。恢复状态包含有效 sigma、encoder、判别器、优化器、RNG 和数据游标，恢复时不会重新校准。不要用正式 LoRA 代替 resume 文件。

---

[文档首页](README.md) · [参数参考](training/parameters.md) · [故障排查](troubleshooting.md)
