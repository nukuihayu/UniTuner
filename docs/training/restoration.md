# 图像修复训练

简体中文 · [English](../en/training/restoration.md)

[文档目录](../README.md) · [训练指南](README.md) · [示例配置](../examples/README.md)

公共训练设置见[训练指南](README.md)，全部配置入口见[示例索引](../examples/README.md)。

Mid-timestep 图像修复把低清图编码的 latent 直接送入固定时间的模型，再做一步恢复；不向低清 latent 额外加噪。模型保留各自的时间单位和速度方向。支持的图像模型可使用 LoRA 或全参数训练，训练后通过 `unituner restore` 修复图片。

<details>
<summary>本页导航</summary>

- [参数与损失速查](#参数与损失速查)
- [局部周期残差约束](#局部周期残差约束)
- [适用模型与后验](#适用模型与后验)
- [数据与训练命令](#数据与训练命令)
- [中间时间配置](#中间时间配置)
- [高分辨率修复](#高分辨率修复)
- [产物与修复推理](#产物与修复推理)
- [独立计算中间时间](#独立计算中间时间)

</details>

## 参数与损失速查

完整的配对数据训练过程见[修复教程](../restoration-tutorial.md)。本页进一步说明时间、后验与参考公式。

| `method.restoration` 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `midtimestep` | `244` | 1 到 998 的整数时间索引 |
| `sigma` | 未设置 | 大于 0、小于 1 的显式噪声比例 |
| `calibrate` | `false` | 训练前估计 sigma，与显式 sigma 互斥 |
| `calibration_pairs` | `32` | 校准最多编码的裁剪对／分块对数 |
| `calibration_sampling` | `"random"` | `random` 每图随机裁剪一次；`tiles` 按 dataset 分辨率逐块统计 |
| `encoder_rank` | `4` | 编码器适配器 rank，0 表示关闭 |
| `posterior` | 模型原生策略 | 在 VAE 支持范围内选择 sample 或 mean |
| `histogram_bins` | `32` | 颜色直方图分箱数，至少为 2 |
| `phase_period`、`phase_cycles` | `16`、`8` | 相位约束周期与每个窗口每轴的完整周期数，均至少为 2 |
| `phase_epsilon` | `0.001` | 相位约束 Charbonnier 平滑尺度，按 `[-1,1]` RGB 计；范围 `1e-12` 到 `1.0` |
| `dino_weights` | 未设置 | 对应损失所需的本地特征网络权重 |
| `dino_widths` | `[192, 384, 768, 1536]` | 默认匹配 DINOv3 ConvNeXt-Large |
| `dino_depths` | `[3, 3, 27, 3]` | DINO 四阶段深度 |

`dino_weights` 支持 `.pt`、`.pth`、`.safetensors`，无需转换或改名。PyTorch 权重通过 `weights_only=True` 读取，必须是平铺的 tensor state dict。`dino_weights` 默认匹配官方 `dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth`；其他尺寸需显式填写对应的 `dino_widths` 和 `dino_depths`。

| `[losses]` 字段 | 默认权重 | 监督内容 | 额外资产 |
| --- | --- | --- | --- |
| `lrr` | `5.0` | LQ latent 与加噪冻结 HQ latent 的均方误差 | 无 |
| `l1` | `0.5` | 恢复图与 HQ 的像素绝对误差 | 无 |
| `dv3d` | `3.0` | DINOv3-ConvNeXt 特征距离 | DINO 权重 |
| `gan` | `0.5` | 对抗图像监督，需判别器更新 | DINO 权重 |
| `histogram` | `0.0` | 归一化 Lab 颜色 CDF 匹配 | 无 |
| `ms_ssim` | `0.0` | 五尺度 RGB 结构相似度：`1 - MS-SSIM` | 无 |
| `phase` | `0.0` | 保护目标纹理的局部周期残差约束，默认关闭 | 无 |

这些是配置默认值，不是适用于所有任务的推荐配方。未使用的损失要显式设为 0；仅省略 DINO 路径，却保留相应默认损失，会导致配置报错。至少启用一种图像输出损失（l1、dv3d、gan、histogram、ms_ssim，或启用有效频带的 `[method.wavelet]`），LRR 与 phase 都不能单独监督完整解码结果。

FDL 已移除。旧配置需删除 `[losses]` 中的 `fdl`，以及 `[method.restoration]` 中的 `vgg_weights`、`vgg_widths`；这些字段不再接受，包括 `fdl = 0.0`。

大数据集启动时需要检查图片和缓存，请留意终端进度，并为图片配对索引和模型权重预留主机内存。

## MS-SSIM

设置 `[losses] ms_ssim = 0.1` 可增加多尺度结构监督，默认值为 `0.0`。权重仅作示例，需用验证图片调整。裁剪宽高都必须大于 160 像素，不需要额外网络权重。

日志中的 `ms_ssim` 是未乘权重的损失，越低越好。需要保留像素和颜色准确性时可与 L1 配合，选择权重时也应检查真实细节。

## 小波 MSE

小波 MSE 使用 `[method.wavelet]`：递归分解输出和 HQ，对每个启用的层与频带计算元素均值 MSE，再按频带系数求和并乘 `weight`，不是把全部系数放在一起取一次均值。`normalize_bands = false` 保留频带的绝对幅度。恢复训练不应用时间步衰减，`timestep_cutoff` 和 `timestep_transition_width` 仅用于流匹配。小波损失不需要外部特征权重。

```toml
[losses]
lrr = 0.0
l1 = 0.0
dv3d = 0.0
gan = 0.0
histogram = 0.0
phase = 0.0

# 小波监督配置。
[method.wavelet]
weight = 0.5
wavelet = "haar"
levels = 3
ll = 0.1
lh = 0.01
hl = 0.01
hh = 0.05
ll_level_threshold = -1
normalize_bands = false
```

这只是配置示例，不是调优后的权重推荐。仅使用小波 MSE 时，必须保留至少一个有效频带。`ll_level_threshold = -1` 只在最后一级计入 LL，细节频带则在每级计入。`wavelet` 指标已包含总权重 `weight`。参见[全部小波参数](parameters.md#methodwavelet)。

## 局部周期残差约束

`losses.phase` 用于抑制恢复图相对 HQ 的周期或网格伪影，默认关闭，不需要额外特征权重。在已有配方中加入：

```toml
[losses]
phase = 4.0

[method.restoration]
phase_period = 16
phase_cycles = 8
phase_epsilon = 0.001
```

训练裁剪宽高都必须至少为 `phase_period * phase_cycles`，本例为 128 像素。周期应根据输出中实际出现的伪影选择，不会按模型自动设置。这里的权重只是起点示例，并非通用推荐。

保留 L1 或感知损失等主要图像监督。用验证图片同时检查重复纹理、真实细节、边缘和颜色；细节受损时降低权重或关闭。该损失不能修正未对齐的训练对。修改有效损失设置后，可能无法接续原来的完整训练状态。

## 适用模型与后验

| 家族 | 适用变体 | 修复训练后验 |
| --- | --- | --- |
| Z-Image | 当前原生变体 | 默认 sample，可选 mean |
| FLUX | Dev | 默认 sample，可选 mean |
| FLUX.2 | Dev、Klein 4B/9B 的原生 base/distilled 组合 | mean |
| FLUX3 | base、distilled | mean |
| MOSS Image | Base、SFT | mean |
| Qwen Image | original | 默认 sample，可选 mean |
| Qwen Image 2.1 | 原生图像模型 | 默认 sample，可选 mean |
| Krea2 | RAW、Turbo | 默认 sample，可选 mean |

`posterior = "sample"` 会在训练中引入 VAE 后验采样随机性；`"mean"` 使用均值。请按表格选择模型支持的值。预览和修复推理使用均值。修改此设置后，需要检查缓存与续训兼容性。

FLUX Kontext/Fill、Qwen edit/layered、Wan 和 MiniMaxH3 不适用此修复方法。

## 数据与训练命令

示例：[FLUX](../examples/flux-restoration.toml)、[FLUX.2](../examples/flux2-restoration.toml)、[FLUX3](../examples/flux3-restoration.toml)、[MOSS](../examples/moss-image-restoration.toml)。示例只启用 LRR 和 L1，不需要下载感知损失权重。`dataset.groups.main.source` 和 `low_quality_source` 按图像文件主体匹配；两目录图像需要对齐。缓存命令提取原生文本，FLUX 同时保留 T5 和 CLIP pooled 特征。FLUX 的辅助 CLIP 资产放在本地 bundle 的 `text_encoder/`，`model.assets` 指定 bundle 根目录，省略时使用 `model.weights` 的父目录。

```bash
unituner cache docs/examples/flux3-restoration.toml --device cuda
unituner train docs/examples/flux3-restoration.toml --device cuda
unituner train docs/examples/flux3-restoration.toml --device cuda --resume runs/flux3-restoration/resume/model.state-00000050.safetensors
```

改为 full 时，设置 `[train] parameter_mode = "full"` 并删除整个 `[train.adapter]`；流匹配训练也使用同样的切换方式。可训练参数保持 FP32；Full 要求浮点基模。适配器可用[量化冻结基模](quantized-base.md)，修复与普通流匹配一样支持 `[execution] precision = "bf16"`，VAE、DINO 和判别器的卷积与线性层均使用 BF16 AMP；可训练主参数和优化器状态沿用普通流匹配的精度策略，损失统计保留 FP32。Full 使用 resident，可组合 block 重算和 CUDA compile；冻结块 H2D 只适用于 adapter。TREAD、REPA、LayerSync 当前只接入流匹配，修复配置会拒绝这些组合。

## 中间时间配置

中间时间使用公共 `method.restoration.midtimestep`，默认 `244`；CLI 显式值覆盖 TOML：

```toml
[method]
kind = "restoration"
[method.restoration]
midtimestep = 244
```

```bash
unituner train train.toml --device cuda --midtimestep 244
```

`midtimestep` 接受 1..998 的整数，仅适用于图像修复。省略 CLI 选项时保留 TOML 值；两者都省略才取244。
`method.restoration.sigma` 默认未指定，可直接给定噪声比例；`calibrate = true` 则按成对数据选择。
选时顺序为：启用校准则使用校准结果，否则使用显式 sigma，最后使用 midtimestep。
`calibrate` 与显式 sigma 仍互斥；CLI 只覆盖 midtimestep 数值，不取消已配置的校准或 sigma 策略。

FLUX.1 的整数表示 999 步移位时间表的反向索引：
`build_schedule(999, (W//16)*(H//16))[-(midtimestep+1)]`，W/H 为数据集配置的像素裁剪尺寸。
因此 FLUX.1 的244 **不等于 sigma=0.244**。其余已接入家族当前使用 `midtimestep/1000`，
保留各自原生时间单位和速度方向；这不是为所有模型重新估计出的最优时间。
## 高分辨率修复

高分辨率训练显存不足时，先减小 batch 或裁剪尺寸，再启用重算：

```toml
[execution]
recompute = "block"
restoration_recompute = "block"
```

重算以额外计算时间换取激活显存。感知和对抗损失也会占用显存，仅启用需要的损失并准备对应权重。详见[执行设置](execution.md)和[精度设置](precision.md)。

更换模型、VAE、分辨率、数据或退化配置后，建议重新校准 sigma。校准结果会保存在训练产物中，续训时沿用该值，不会重新校准。

## 产物与修复推理

最终输出包括 denoiser 的 full/adapter 文件、可选 encoder adapter 文件和完整训练 checkpoint。发布的模型/adapter 和 encoder 都记录 sigma/posterior；恢复推理检查各产物的 sigma 一致性。Full 推理把 `model.weights` 指向训练后的模型；LoRA 推理保留底模，设置 `[adapter]` 与 `[adapter.config]`。`encoder_weights` 指向训练导出的 encoder 文件。一步推理使用后验均值，保留输入原尺寸。

## 独立计算中间时间

```bash
unituner mid-timestep docs/examples/restoration-calibration.toml --device cuda
```

可复制配置见[独立校准示例](../examples/restoration-calibration.toml)。Full 配方见[修复全参数示例](../examples/restoration-full.toml)。
命令只读取图像与本地 `model.vae`，不加载 DiT、文本编码器、感知损失或优化器，不需要预先生成缓存。
沿用数据集的对齐目录、合成退化、裁剪尺寸和 `train.seed`；默认取前32个唯一图像对，使用
`[method.restoration] calibrate = true`、`calibration_pairs = 128` 可扩大计算样本。
结果打印 JSON：`midtimestep` 是索引，`sigma` 是实际噪声比例，另含 `mean_snr_error`、`pairs`、`posterior`、`sampling`。
可以把结果中的 sigma 写入 `method.restoration.sigma` 并关闭 calibrate，或启用 calibrate 在训练前自动估计一次。
训练 TOML 可直接用于独立命令，合成数据会读取 `[dataset.groups.<名称>.degradation]` 的完整配方，不需要先保存 LQ 图像。先按 `dataset.defaults.resolution` 或组内覆盖的 `[宽, 高]` 取块，再执行与训练相同的退化，最后分别编码 LQ/HQ 计算 SNR。真实配对图必须已经同尺寸对齐，不会自动拉伸或缩放。修复使用固定裁剪尺寸，不采用文生图的宽高比分桶。

```toml
[method.restoration]
calibrate = true
calibration_pairs = 128
calibration_sampling = "tiles"

[dataset.defaults]
resolution = [512, 512]

[dataset.groups.main]
source = "/data/hq"
# 训练时仍需其余 dataset 字段；独立命令不需要 cache_directory。

[dataset.groups.main.degradation]
enabled = true
```

`tiles` 按图像顺序、块的行列顺序处理；不足一个块的小图反射补齐，最后一行／列贴边取完整块，必要时与前块重叠。每块独立执行配置的翻转与模拟退化，并以该块的 latent 统计 SNR。`calibration_pairs` 此时是**总块数上限**，不是原图数量；块等权，大图贡献更多样本，达到上限后后续图像不会处理。若希望更多原图参与，使用默认 `random`，或增大上限。随机种子固定时结果可复现，校准不会推进训练 RNG。

独立的最小校准 TOML 将 `calibration_sampling`、`calibration_pairs`、`posterior` 放在顶层；训练 TOML 放在 `[method.restoration]`。所有组仍须共享同一个分辨率，避免混合 FLUX 分辨率相关时间网格。分块是校准样本提取，不是把整图 VAE 编码切块后拼接，也不会改变正式训练的随机裁剪策略。


修复目标采用 [OMGSR 方法](https://github.com/wuer5/OMGSR)。项目损失默认值见[参数手册](parameters.md#losses)；使用不同配方时显式填写系数。


---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
