# 图像修复训练

简体中文 · [English](../en/training/restoration.md)

[文档目录](../README.md) · [训练指南](README.md) · [示例配置](../examples/README.md)

公共训练设置见[训练指南](README.md)，全部配置入口见[示例索引](../examples/README.md)。

Mid-timestep 图像修复把低清图编码的 latent 直接送入固定时间的模型，再做一步恢复；不向低清 latent 额外加噪。模型保留各自的时间单位和速度方向。八个图像家族均支持 LoRA 和 full，训练、预览、checkpoint 恢复和 `unitunner restore` 使用同一公式。

<details>
<summary>本页导航</summary>

- [参数与损失速查](#参数与损失速查)
- [局部周期残差约束](#局部周期残差约束)
- [适用模型与后验](#适用模型与后验)
- [数据与训练命令](#数据与训练命令)
- [中间时间配置](#中间时间配置)
- [与 OMGSR 论文的对应边界](#与-omgsr-论文的对应边界)
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
| `dino_weights`、`vgg_weights` | 未设置 | 对应损失所需的本地特征网络权重 |
| `dino_widths` | `[192, 384, 768, 1536]` | 默认匹配 DINOv3 ConvNeXt-Large |
| `dino_depths` | `[3, 3, 27, 3]` | DINO 四阶段深度 |

`dino_weights` 和 `vgg_weights` 均支持 `.pt`、`.pth`、`.safetensors`，无需转换或改名。PyTorch 权重通过 `weights_only=True` 读取，必须是平铺的 tensor state dict。`dino_weights` 默认匹配官方 `dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth`；其他尺寸需显式填写对应的 `dino_widths` 和 `dino_depths`。VGG19 可直接使用官方 `vgg19-dcbb9e9d.pth`（自动排除已知分类头），也支持仅特征层的权重；未知键和缺失的特征权重仍会报错。

| `[losses]` 字段 | 默认权重 | 监督内容 | 额外资产 |
| --- | --- | --- | --- |
| `lrr` | `5.0` | LQ latent 与加噪冻结 HQ latent 的均方误差 | 无 |
| `l1` | `0.5` | 恢复图与 HQ 的像素绝对误差 | 无 |
| `dv3d` | `3.0` | DINOv3-ConvNeXt 特征距离 | DINO 权重 |
| `gan` | `0.5` | 对抗图像监督，需判别器更新 | DINO 权重 |
| `fdl` | `0.5` | 特征频率分布 | VGG19 权重 |
| `histogram` | `0.0` | 归一化 Lab 颜色 CDF 匹配 | 无 |
| `ms_ssim` | `0.0` | 五尺度 RGB 结构相似度：`1 - MS-SSIM` | 无 |
| `phase` | `0.0` | 保护目标纹理的局部周期残差约束，默认关闭 | 无 |

这是 schema 默认值，不是适用于所有任务的推荐配方。未使用的损失要显式设为 0；仅省略 DINO／VGG 路径，却保留相应默认损失，会导致配置报错。至少启用一种图像输出损失（l1、dv3d、gan、fdl、histogram、ms_ssim，或启用有效频带的 `[method.wavelet]`），LRR 与 phase 都不能单独监督完整解码结果。

大数据集启动索引时，源图片、caption 和缓存文件逐项校验，缓存只读取头部元数据，不载入 latent／文本张量。数据集摘要逐条计算，避免保留额外的逐图身份字典和完整 JSON 副本；必要的文件配对索引仍随图片数量增长。该优化不改变缓存格式或续训摘要，无需重建缓存。终端会显示逐图校验进度。

Linux 的系统／容器 RAM 统计也可能包含文件页缓存。修复索引在完整哈希每个 HQ、LQ 和缓存文件后，会自动用 `POSIX_FADV_DONTNEED` 提示内核释放该文件的已读页面，减少一次性扫描积累的页缓存；无需额外配置或管理员权限。此提示由内核／文件系统尽力执行，不保证释放仍被映射、使用或尚未写回的页面。不支持此接口的平台照常校验；不会清空全系统缓存，也不会跳过内容校验。后续训练读取图片时可能需要重新从磁盘读取。

以上策略也用于修复训练的模型／VAE／特征权重摘要扫描，包括分片权重，摘要和续训兼容性不变。DINO／VGG 的 ZIP 格式 PyTorch 权重在加载到 GPU 时采用只读源文件的私有内存映射，避免先把完整 checkpoint 复制到匿名 CPU 内存；旧版 pickle 文件保留普通读取方式。映射触及的文件页面仍可能计入容器 RAM。

八个图像家族（Z-Image、FLUX.1、FLUX.2、FLUX3、MOSS、Qwen Image、Qwen Image2.1、Krea2）共用以下内存优化：训练／预览只读取缓存中的文本条件，FLUX.1 额外读取 pooled 特征，保留 Krea2 的选层文本布局，不搬运未使用的缓存 latent／参考图；RGB 解码避免额外的整图 PIL 副本，只在确实需要时做反射填充，合成退化只创建一份 HQ 裁剪；batch=1 不额外堆叠图像，CPU 裁剪在设备搬运后及时释放；生成器反向结束后，清理其临时 latent 和计算图引用，再进入判别器阶段。这些优化无需修改配置，不改变裁剪／退化随机数、损失权重或模型公式；不会降低模型、优化器和必要激活本身的存储需求。

## MS-SSIM

在 `[losses]` 中设置 `ms_ssim = 0.1` 即可启用（仅为配置示例，不是实测推荐权重）。默认 `0.0` 时跳过计算，并保持旧配置的续训标识；启用后权重与数值策略纳入续训标识。日志中的 `ms_ssim` 是未乘权重、已 detach 的损失。

损失为按样本和 RGB 通道平均的 `1 - MS-SSIM`。使用五尺度权重 `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`、11×11 高斯窗口（sigma=1.5）、valid 卷积和 K1=0.01/K2=0.03。尺度间采用 2× 平均池化，奇数尺寸按奇偶性补边。图像宽高都必须大于 160 像素，配置阶段会拒绝过小的训练裁剪，不会自动减少尺度数。

BF16/FP16 训练时，统计和归约仍采用 FP32。将 nominal `[-1,1]` RGB 映射到 `[0,1]`，不裁断解码器的越界输出；各尺度分数在分数次幂前限制到 `[1e-6,1]`，避免负相似度产生非有限梯度，低于下限的分数梯度会饱和。HQ 停止梯度。不需要额外特征网络或依赖。

MS-SSIM 监督多尺度结构，不是周期伪影专用损失。可与 L1 配合保留像素／颜色监督；配置也允许它单独作为图像输出损失。数值验证不代表真实恢复质量一定改善，也不能保证消除网格，需要在保留验证图片上选择权重。

## 小波 MSE

小波 MSE 使用 `[method.wavelet]`：递归分解输出和 HQ，对每个启用的层与频带计算元素均值 MSE，再按频带系数求和并乘 `weight`，不是把全部系数放在一起取一次均值。`normalize_bands = false` 保留频带的绝对幅度。恢复训练不应用时间步衰减，`timestep_cutoff` 和 `timestep_transition_width` 仅用于流匹配。小波损失不需要外部特征权重。

```toml
[losses]
lrr = 0.0
l1 = 0.0
dv3d = 0.0
gan = 0.0
fdl = 0.0
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

`losses.phase` 是默认关闭的辅助损失，抑制恢复图相对 HQ 的重复相位偏差。最终实现同时保护局部亮度平面和目标纹理的对比度，避免把它们全部当成网格错误。

每个窗口对残差 `e = generated - target` 和目标分别拟合并扣除平面，得到 `e_t`、`y_t`，再计算：

```text
beta = mean(e_t * y_t) / (mean(y_t²) + 1e-6)
r = e_t - beta * y_t
m = 按坐标模 phase_period 分组的 r 均值，再扣除相位均值
phase = mean(sqrt(m² + phase_epsilon²) - phase_epsilon)
```

`beta` 是带正则化的局部对比度拟合，`1e-6` 是按 `[-1,1]` RGB 计的方差正则项，防止平坦目标产生不稳定系数。系数参与正常反向；只有目标停止梯度。统计使用 FP32，不增加特征网络前向，不改模型推理。这个与目标有关的线性变换加稳健惩罚，不是单个正交投影。

正确重建的真实周期纹理不受罚；与目标纹理成比例的局部误差会减轻处罚，由 L1、感知等主损失继续监督。错位边缘和细节仍可能受影响。输出空间的亮度保护也不保证训练后亮度完全不变，因为共享网络参数会耦合不同误差方向。

在已有配置中合入以下字段即可启用，其他损失和模型设置沿用原配方：

```toml
[losses]
phase = 4.0

[method.restoration]
phase_period = 16
phase_cycles = 8
phase_epsilon = 0.001
```

窗口为 `16 × 8 = 128` 像素，每幅训练图宽高都必须至少为 128。内部窗口不重叠；末端不足一个窗口时追加贴边完整窗口，与相邻窗口重叠。不填充、不丢弃边缘，每个窗口等权。周期 16 是 Z-Image 的候选值，不会按模型自动选择；应根据原始输出测得的周期设置，勿盲目叠加 2/4/8/16。此损失不能单独监督完整图像，也不能修复未对齐的训练对。

真实照片对照使用 Z-Image Turbo、官方 DINOv3 ConvNeXt Large、512 裁剪，80 张照片训练、20 张独立照片验证（40 裁剪），由项目原生流程合成退化。每组 300 次更新、累积 4、学习率 `5e-5`，LoRA rank/alpha 8/8、编码器 rank 4、显式 sigma 0.121、BF16、空文本条件，无 EMA。LRR/Dv3D/L1/GAN 为 5/3/0.5/0.5，FDL 与 histogram 为 0；同种子内初始化、样本顺序和噪声完全配对。

| 种子 | 局部相位 RMS 变化 | PSNR（基线 → 启用） | Dv3D（基线 → 启用，越低越好） | SSIM（基线 → 启用） |
| --- | --- | --- | --- | --- |
| 123 | −26.1% | 26.495 → 26.734 dB | 0.12546 → 0.12450 | 0.6615 → 0.6810 |
| 456 | −9.2% | 27.264 → 27.288 dB | 0.12235 → 0.12079 | 0.69944 → 0.69951 |

两次试验中，20 张验证照片的局部相位误差均下降；平滑区域平均误差分别下降 26.4% 和 11.2%。但并非逐图各项都改善：第二个种子的边缘误差增加约 1.3%，SSIM 平均基本持平，个别原图的平均 PSNR 下降约 1 dB。仅去除亮度平面的早期版本在第二个种子上损失约 0.39 dB，因此最终只保留上述目标纹理保护实现。单纯把 L1 从 0.5 提至 1.0，也没有稳定优于各自基线，未修改默认 L1。

这只是两个种子、训练初期的探索性对照，使用真实照片的合成退化；验证集参与方案筛选，不是独立最终测试集或官方 DIV2K 基准。它不保证消除用户数据中的全部伪影，也未证明长期收敛质量普遍提升。保持默认关闭，4.0 只是经测试的起点；开启后应同时观察重复纹理、真实细节、边缘与颜色。关闭 phase 保留此前的续训身份；启用或改变有效损失参数会改变身份，不能静默接续原损失的完整训练状态。

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

启用 LRR 时，共享修复步骤按“冻结 HQ 后验编码 → LRR 高斯噪声 → 可训练 LQ 后验编码”执行。采样型 VAE 的后验噪声按逻辑形状生成，避免 channels-last 改变随机值与元素的对应关系；均值型 VAE 不增加后验采样。该顺序对齐 Trainer 的有效 LRR 路径，不改变各模型原生 latent 归一化、时间或预测符号。

后验采样增加训练随机性，不能视为确定的质量或速度提升；显式 mean 仍可用于确定性变体。关闭 LRR 时跳过无用 HQ 编码及其随机抽取，因此不承诺与 Trainer 保持逐步 RNG 一致。旧训练的后验/随机顺序变化不允许静默续训。预览和修复推理仍取均值。

FLUX Kontext/Fill 和 Qwen edit 系列不在修复范围内；Qwen layered 需要独立图层监督，当前修复入口仅支持 original。Wan、MiniMaxH3 是视频/音视频流程，不接入此方法。FLUX.2/FLUX3/MOSS 共用原生 packed-BN VAE 的均值编码，不提供未经原生实现验证的 posterior sample。

## 数据与训练命令

示例：[FLUX](../examples/flux-restoration.toml)、[FLUX.2](../examples/flux2-restoration.toml)、[FLUX3](../examples/flux3-restoration.toml)、[MOSS](../examples/moss-image-restoration.toml)。示例只启用 LRR 和 L1，不需要下载感知损失权重。`dataset.groups.main.source` 和 `low_quality_source` 按图像文件主体匹配；两目录图像需要对齐。缓存命令提取原生文本，FLUX 同时保留 T5 和 CLIP pooled 特征。FLUX 的辅助 CLIP 资产放在本地 bundle 的 `text_encoder/`，`model.assets` 指定 bundle 根目录，省略时使用 `model.weights` 的父目录。

```bash
unitunner cache docs/examples/flux3-restoration.toml --device cuda
unitunner train docs/examples/flux3-restoration.toml --device cuda
unitunner train docs/examples/flux3-restoration.toml --device cuda --resume runs/flux3-restoration/resume/model.state-00000050.safetensors
```

改为 full 时，设置 `[train] parameter_mode = "full"` 并删除整个 `[train.adapter]`；流匹配训练也使用同样的切换方式。可训练参数保持 FP32；Full 要求浮点基模。适配器可用[量化冻结基模](quantized-base.md)，修复与普通流匹配一样支持 `[execution] precision = "bf16"`，VAE、DINO、VGG 和判别器的卷积与线性层均使用 BF16 AMP；可训练主参数和优化器状态沿用普通流匹配的精度策略，FFT 与损失统计保留 FP32。Full 使用 resident，可组合 block 重算和 CUDA compile；冻结块 H2D 只适用于 adapter。TREAD、REPA、LayerSync 当前只接入流匹配，修复配置会拒绝这些组合。

## 中间时间配置

中间时间使用公共 `method.restoration.midtimestep`，默认 `244`；CLI 显式值覆盖 TOML：

```toml
[method]
kind = "restoration"
[method.restoration]
midtimestep = 244
```

```bash
unitunner train train.toml --device cuda --midtimestep 244
```

`midtimestep` 接受 1..998 的整数，仅适用于图像修复。省略 CLI 选项时保留 TOML 值；两者都省略才取244。
`method.restoration.sigma` 默认未指定，可直接给定噪声比例；`calibrate = true` 则按成对数据选择。
不再根据 midtimestep 是否等于244产生限制。选时顺序为：启用校准则使用校准结果，否则使用显式 sigma，最后使用 midtimestep。
`calibrate` 与显式 sigma 仍互斥；CLI 只覆盖 midtimestep 数值，不取消已配置的校准或 sigma 策略。

FLUX.1 的整数表示 Trainer 原生999步时间表的反向索引：
`build_schedule(999, (W//16)*(H//16))[-(midtimestep+1)]`，W/H 为数据集配置的像素裁剪尺寸。
因此 FLUX.1 的244 **不等于 sigma=0.244**。其余已接入家族当前使用 `midtimestep/1000`，
保留各自原生时间单位和速度方向；这不是为所有模型重新估计出的最优时间。
时间换算保留原生运算顺序，例如 FLUX 系列的 `full(sigma*1000)/1000`。
FLUX.1 修复使用固定 guidance embedding=1.0，与 Trainer OMGSR-F 一致，不改变流匹配或普通采样的默认值。
FLUX3 distilled 的训练目标时间与源时间相同，沿用原 finetuning 入口。

## 与 OMGSR 论文的对应边界

依据 [OMGSR v2 §3 与补充材料 §6](https://arxiv.org/html/2508.08227v2)：

- LQ latent 直接输入，不额外加噪；训练 VAE encoder adapter 以缩小表示差距。
- FM 的 LRR 目标为 `(1-sigma)*encode(HQ) + sigma*noise`，HQ 分支冻结；使用均方误差。
- noise-minus-data 模型恢复为 `low - sigma*velocity`；Z-Image 的符号在模型边界处理。
- 校准使用公式18/19的逐图 SNR 差绝对值的平均目标，FLUX.1 的候选取同一移位网格。
- 冻结 DINOv3-ConvNeXt 的前三阶段用于 Dv3CD 与多级判别，生成图像分支保留梯度。

`[execution] recompute = "block"` 同时对 DINO 的降采样层和 ConvNeXt 块启用激活重算，减少 Dv3CD 与生成器 GAN 分支的激活显存，以额外反向重算为代价。HQ 特征和判别器更新中的冻结骨干不需要图像梯度，不执行重算；判别头也不重算，以避免重复更新 spectral normalization 状态。设置 `recompute = "none"` 可关闭这项时间与显存交换。 DINO 通道归一化使用原生 LayerNorm，BF16 统计以 FP32 累积；与逐元素 BF16 归一化相比允许小幅数值差异。

启用 `recompute = "block"` 时，图像恢复还会重算 VAE 的无状态残差、空间注意力及适用的采样块，降低高分辨率训练的激活显存；VAE 保持 eval，解码器权重仍冻结，编码器 LoRA 梯度保留。Z-Image、FLUX 家族和 Qwen Image 2.1 支持残差块重算；Qwen Image/Krea2 的时间历史块保持原路径，仅重算无状态空间注意力。重算增加计算耗时，且不能消除大卷积工作区或 DiT 注意力的显存需求。

训练 Inductor 编译按图设置 `triton.cudagraphs=false` 和 `epilogue_fusion=true`，避免 CUDA Graph 私有内存池并保留逐元素融合；不修改进程全局配置。当前编译入口针对 DiT，不会自动编译 VAE/DINO。GEMM 最大调优、坐标下降调优和 1×1 卷积转矩阵乘不作为默认优化；它们的收益依赖具体图与硬件，可能增加首次编译时间。

论文的244来自 FLUX.1-dev、1K分辨率；SD2.1-base 的结果是273。Trainer 的 Z-Image 默认121，
也不是论文对 Z-Image 的结论。本项目按配置约定统一起始默认244；换模型、分辨率、VAE、数据或退化分布后，
应使用对应数据的校准结果，不宣称默认244普适最优。DDPM 的公式7使用 alpha-bar 重建，
不能套用本项目 FM 的 Euler 恢复公式；当前入口没有新增 DDPM 支持。

默认配方的 LRR/Dv3D/L1/GAN/FDL 权重为5/3/0.5/0.5/0.5，直方图默认0。
FDL、直方图与 phase 是额外训练配方；不宣称它们是论文原始 OMGSR 损失或已复现论文质量。

FDL 使用正交归一化 FFT（`norm="ortho"`），按特征图尺寸规范幅值尺度；相位项、256 个投影及分层求和保持不变。这是原始未归一化 FDL 的变体，不保证损失在 0～1。旧版 FDL 数值不能直接比较，继续训练时应重新评估 `losses.fdl` 权重。

校准使用未适配 VAE，按修复的 posterior 策略编码；FLUX 默认 sample，与官方一致。
后验噪声使用隔离、固定种子的 RNG，校准前后不改变训练 RNG；显式 mean 可做确定性均值变体。
当前网格排除0/1端点，避免无定义 SNR；逐图平均符合论文公式19，而不是对一个批次混合求 SNR。
最终有效 sigma 保存进 checkpoint 和训练产物，续训不重新校准，推理读取相同值。

## 产物与修复推理

最终输出包括 denoiser 的 full/adapter 文件、可选 encoder adapter 文件和完整训练 checkpoint。发布的模型/adapter 和 encoder 都记录 sigma/posterior；恢复推理检查各产物的 sigma 一致性。Full 推理把 `model.weights` 指向训练后的模型；LoRA 推理保留底模，设置 `[adapter]` 与 `[adapter.config]`。`encoder_weights` 指向训练导出的 encoder 文件。一步推理使用后验均值，保留输入原尺寸。

## 独立计算中间时间

```bash
unitunner mid-timestep docs/examples/restoration-calibration.toml --device cuda
```

可复制配置见[独立校准示例](../examples/restoration-calibration.toml)。Full 配方见[修复全参数示例](../examples/restoration-full.toml)。
命令只读取图像与本地 `model.vae`，不加载 DiT、文本编码器、感知损失或优化器，不需要预先生成缓存。
沿用数据集的对齐目录、合成退化、裁剪尺寸和 `train.seed`；默认取前32个唯一图像对，使用
`[method.restoration] calibrate = true`、`calibration_pairs = 128` 可扩大计算样本。
结果打印 JSON：`midtimestep` 是索引，`sigma` 是实际噪声比例，另含 `mean_snr_error`、`pairs`、`posterior`、`sampling`。
可以把结果中的 sigma 写入 `method.restoration.sigma` 并关闭 calibrate，或启用 calibrate 在训练前自动估计一次。
直接复制非FLUX结果时，使用输出的 sigma 可以保留候选网格 FP32 舍入结果；整数索引本身不携带这些舍入位。

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

独立命令与训练前校准共用 `estimate_mid_timestep`，流式累积公式18/19的统计量，不保存全部 latent；
计算开销为每个图像对各两次 VAE 编码加一组向量化候选误差，无需对每个 timestep 重复跑网络。
支持 Z-Image、FLUX.1、FLUX.2、FLUX3、MOSS、Qwen Image、Qwen Image2.1、Krea2 的当前修复变体。
前者按各自 VAE 归一化后计算；FLUX.1 使用其移位时间表，其他当前训练入口使用 t/1000。
新增有不同训练时间表的架构需提供对应候选映射；DDPM 的 alpha-bar 公式不被冒充为 FM sigma。

修复目标采用 [OMGSR 方法](https://github.com/wuer5/OMGSR)。项目损失默认值见[参数手册](parameters.md#losses)；使用不同配方时显式填写系数。

FDL 保留全部 256 个投影以及完整空间排序，每次仅处理 8 个投影通道。反向重算投影与排序，不长期保存全部排序索引；VGG 生成图分支也使用激活重算，以计算时间换取显存。无需增加配置。


---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
