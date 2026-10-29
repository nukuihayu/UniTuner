# 数据集填写教程

[English](../en/training/datasets.md) · [训练教程](README.md) · [参数手册](parameters.md)

<details>
<summary>本页导航</summary>

- [一个 TOML 放模型、数据和训练参数](#一个-toml-放模型数据和训练参数)
- [多组数据与 defaults](#多组数据与-defaults)
- [同名 txt 与统一 prompt](#同名-txt-与统一-prompt)
- [尺寸与长宽比分桶](#尺寸与长宽比分桶)
- [真实 LQ/HQ 配对修复](#真实-lqhq-配对修复)
- [编辑、参考图、掩码和图层](#编辑参考图掩码和图层)
- [视频数据](#视频数据)
- [哪些变化需要重建缓存](#哪些变化需要重建缓存)

</details>

## 一个 TOML 放模型、数据和训练参数

不必维护独立 dataset TOML。在同一个文件中填写：

```toml
[model]
family = "z-image"
variant = "base"
weights = "/models/z-image.safetensors"
vae = "/models/ae.safetensors"
text_encoder = "/models/qwen3"

[dataset.groups.main]
kind = "image"
source = "/data/images"
cache_directory = "/data/cache/main"
resolution = [1024, 1024]
batch_size = 1

[train]
parameter_mode = "adapter"
learning_rate = 1e-4
max_updates = 1000

[train.adapter]
algorithm = "lora"
rank = 32
alpha = 32.0
```

这份最小训练配置可通过 validate；执行 cache 还需确认模型文件、tokenizer 和图片存在。训练方法默认流匹配；训练精度默认按底模解析。

## 多组数据与 defaults

```toml
[dataset.defaults]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true
caption_extension = ".txt"

[dataset.groups.portraits]
kind = "image"
source = "/data/portraits"
cache_directory = "/data/cache/portraits"
repeats = 3

[dataset.groups.scenes]
kind = "image"
source = "/data/scenes"
cache_directory = "/data/cache/scenes"
resolution = [768, 512]
batch_size = 2
repeats = 1
```

defaults 只放共享采样设置：resolution、batch_size、repeats、enable_bucket、bucket_no_upscale、prompt、caption_extension。路径、kind 和条件放在每个组中。组内字段覆盖 defaults。

portraits 的每张图片在数据轮转中出现 3 次，scenes 出现 1 次；组有不同图片数时，组频率还取决于图片数和分桶批量。repeats 不表示固定抽样百分比，也不改变 max_updates。不同 batch_size 混合时，每次更新的实际样本数可能不同。

组名用 main、portraits 等 ASCII 标识符；不能含横杠、空格或以数字开头。每组必须有独立 cache_directory，即使两个字符串写法不同但解析到同一目录也不允许。

## 同名 txt 与统一 prompt

```text
/data/portraits/
├── a.png
├── a.txt
├── b.jpg
└── b.txt
```

| 需要的文本策略 | 如何填写 |
| --- | --- |
| 每张图不同文本 | 不填 prompt，读取同名 caption_extension 文件 |
| 所有图片相同文本 | 在 defaults 或某个 group 填 prompt="统一描述" |
| 所有图片空文本 | 明确填 prompt=""，已有 txt 也会被覆盖 |
| 大部分组统一，某一组不同 | defaults.prompt 设置公共文本，再用 group.prompt 覆盖 |

TOML 没有 null。不要写 prompt=null，也不要把 "null" 当作删除默认值。若混合“统一文本组”和“逐图文本组”，将 prompt 仅写在需要统一文本的组中。

图片目录读取直属图片，不递归扫描子目录。新增图片或修改文本后重新 cache；不需要为全局 prompt 生成重复 txt。

## 尺寸与长宽比分桶

resolution=[宽,高]，单位像素。开启 enable_bucket 时按目标面积与原图宽高比选桶，然后缩放裁剪；关闭时使用固定分辨率裁剪。bucket_no_upscale=true 避免为了桶尺寸放大小图。

多尺度训练可以在 defaults 或单组中填写尺寸列表：

```toml
[dataset.groups.main]
source = "./images"
cache_directory = "./cache/main"
resolution = [[1024, 1024], [768, 768], [512, 512]]
batch_size = 1
enable_bucket = true
prompt = "a portrait photo"
```

每张图在每个尺度各生成一份训练样本，不是在这些尺寸中随机选一个。每档分别分桶；关闭分桶则按每个指定宽高缩放裁剪。三档的缓存分别写入 `cache/main/1024x1024`、`768x768`、`512x512`，训练按独立数据组混合，组名为 `main__1024x1024` 等。每档继承本组 repeats、batch_size、prompt 和参考图；默认按样本/批次混合，不因像素数增加采样权重。各档不会拼进同一批次，max_updates 仍是整个训练的更新总数。

单个 `[1024,1024]` 继续直接使用 cache_directory；嵌套列表即使只有一档也使用尺寸子目录。列表必须非空、尺寸为正整数且不能重复；修改尺寸列表后重新 cache。修复训练及 mid-timestep 校准目前仍要求统一的单一分辨率，多尺度列表用于生成/编辑训练。

一般图像家族使用 16 对齐尺寸；Qwen Image 2.1、H3 使用 32 对齐。图像尺寸越大，激活和注意力开销越高。先减 batch_size，再比较重算或降低 resolution。

对齐条件图使用目标桶。原始构图没有对齐的图像，不会因为文件名相同就自动配准。配对修复前应先检查 LQ/HQ 的位置和尺寸。

## 真实 LQ/HQ 配对修复

```toml
[dataset.groups.main]
kind = "restoration"
source = "/data/target"
low_quality_source = "/data/control"
cache_directory = "/data/cache/restoration"
resolution = [1024, 1024]
batch_size = 1
prompt = ""
horizontal_flip_probability = 0.0
```

source 是 HQ/target，low_quality_source 是输入 LQ/control，不要反填。两目录按文件主体配对，例如 target/a.png 对 control/a.jpg。成对水平翻转同时作用于 LQ/HQ，方向敏感时设 0。

已有真实图片时不填写 degradation。只有需要主动构造低质量输入时才使用 `[dataset.groups.NAME.degradation] enabled=true`，并删除同组 low_quality_source。两种来源可以放在不同组中，见 [restoration-mixed](../examples/restoration-mixed.toml)。所有修复组必须使用相同 resolution，以匹配校准尺度。

### 加速启动校验

图像缓存和恢复数据默认使用 4 个线程并行执行完整校验，包括图片／缓存哈希、caption 匹配和缓存头检查。按输入顺序汇总结果，不改变样本顺序或续训标识。最多排队 2 倍线程数的任务，不加载张量正文，避免为整个数据集堆积任务；扫描后继续请求系统回收文件页缓存。

```toml
[train]
dataset_validation_workers = 4
skip_dataset_validation = false
```

NVMe／SSD 可比较 4 和 8 的启动时间；机械盘或带宽受限的存储可设置 1，避免随机读取竞争。线程数范围为 1～32。实际提速取决于存储带宽和哈希吞吐，不减少需要校验的字节数。

### 跳过启动时的完整数据校验

数据和缓存已验证且保持不变时，可将以下字段合并到现有 `[train]`：

```toml
[train]
skip_dataset_validation = true
```

默认 `false`。此参数适用于图像缓存训练和图像恢复训练：跳过缓存文件的完整哈希与记录预检；恢复训练还跳过 HQ 图片、caption 的读取和内容匹配，改用缓存头中的来源标识。仍读取清单与必要的缓存头，检查模型类型、样本数、配对和分桶形状；按需加载数据时仍执行格式检查。真实 LQ 配对仍计算 LQ 哈希，以保留原有续训标识。合成退化无需这一轮图片内容扫描。此选项不影响视频／语言模型流程、缓存生成或模型权重哈希。

开关本身不改变续训标识，未改动的数据可直接续训。开启后不会发现被修改的 HQ、caption 或缓存内容；更换数据或重建缓存后应关闭此项完成校验。

### 合成超分退化

以 [HYPIR 配置与实现](https://github.com/XPixelGroup/HYPIR/tree/b61d107c6cef38f01a93c7833558869731cfa8c1/HYPIR/dataset) 为基础，保留不同倍率适配、真实 JPEG 和末端 10% grain。

1. 裁剪/水平翻转后，默认直接使用原始 HQ，USM 关闭。显式 use_usm=true 可启用锐化（weight=0.5，threshold=10，参考核 51）；此时 sharpen_target=true 会同时锐化监督 GT 和退化源，设 false 则保留原始 HQ 监督。输入张量不修改。
2. 第一阶段 blur/sinc → resize → Gaussian/Poisson → JPEG。
3. 第二阶段可选 blur/sinc → resize → Gaussian/Poisson。
4. 等概率执行「回最终 LQ 尺寸 + final sinc → JPEG」或「JPEG → 回最终 LQ 尺寸 + final sinc」。
5. bicubic 回 HQ 尺寸，**以 10% 概率只添加一次 grain**，最后裁剪、量化到 8-bit。grain 不参与前两阶段噪声，不再被放大或压缩。

```toml
[dataset.groups.synthetic.degradation]
enabled = true
scales = [2.0, 4.0, 6.0, 8.0]
reference_resolution = 512
resolution_exponent = 0.5
use_usm = false
sharpen_target = true

[dataset.groups.synthetic.degradation.final_grain]
probability = 0.1
sigma = [1.0, 8.0]
correlation = [0.3, 0.8]
```

HYPIR 默认输入 512、x4、固定像素核。本项目空间适配是额外扩展：只计算一次 `k=clamp((HQ短边/reference_resolution)^resolution_exponent,0.5,2)`。256/512/1024/1280/2048 对应约 0.707/1/1.414/1.581/2，作用于 sigma、核支撑和 sinc 逆截止频率；不缩放噪声幅度和 JPEG 的 8×8 块。

高倍率适配固定在管道内部：`r=min(1,4/倍率)`，第二阶段向下 resize 抽样值变为 `s'=1-(1-s)*r`，x4/x6/x8 下限为 0.30/0.533/0.65；final sinc 空间系数为 `max(0.5,k*r)`。第二阶段 blur 在其 resize 之前，不按中间网格重复缩小 sigma。适配不再暴露配置开关。

采集噪声沿用 HYPIR 的采样和裁剪语义，但默认上限适度收窄：在每阶段实际 resize 后的网格直接使用抽到的 Gaussian sigma / Poisson scale，不乘分辨率、倍率、像素面积或衰减指数。此前的网格幅度衰减已删除。第一阶段 Gaussian sigma [1,25]、Poisson scale [0.05,2.5]；第二阶段分别为 [1,18]、[0.05,1.8]。第二阶段在 LQ 网格直接加噪，上限收窄更多，以减少强采集噪声叠加后的过重尾部；高倍率下放大的采集噪声仍可比末端 grain 粗。trace 的 parameter 就是实际使用的采样参数，另记录 gray 和 clip 前 noise_rms_255；编码和滤波仍会改变最终可见 RMS。旧适配开关仍已删除，旧 TOML 对应字段需要删除。

基础核支撑 7～21，sigma [0.2,3]/[0.2,1.5]，六种混合核概率 [0.45,0.25,0.12,0.03,0.12,0.03]。第二阶段 blur 80%；启用滤波时各有 10% 替换成 sinc，final sinc 80%。阶段 cutoff 为 π/3～π（基础核<13）或 π/5～π，final 为 π/3～π，不加窗。HYPIR 把小核补零到 21 只是批处理要求，不增加有效支撑。

两阶段采集噪声必加，Gaussian/Poisson 各 50%，灰噪声 40%；sigma [1,25]/[1,18]，Poisson scale [0.05,2.5]/[0.05,1.8]。加噪后 clip、不立即量化；仅 Poisson 采样信号预先量化并统计颜色数。末端 grain 为相关亮度噪声，RMS [1,8]/255，相关半径 [0.3,0.8] 个**输出像素**，不乘空间系数或倍率。整体启用率就是 10%。

参数取舍参考：[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/options/train_realesrgan_x4plus.yml) 和 [HYPIR](https://github.com/XPixelGroup/HYPIR/blob/b61d107c6cef38f01a93c7833558869731cfa8c1/configs/sd2_train.yaml) 原始范围为 sigma [1,30]/[1,25]、Poisson [0.05,3]/[0.05,2.5]；[DiffBIR v2.1](https://github.com/XPixelGroup/DiffBIR/blob/main/configs/train/train_stage2_v2.1.yaml) 也沿用此范围。[ResShift](https://github.com/zsyOAOA/ResShift/blob/master/configs/realsr_swinunet_realesrgan256.yaml) 以 50% 概率执行第二阶段；[DiffBIR 早期配置](https://github.com/XPixelGroup/DiffBIR/blob/main/configs/train/train_stage1.yaml) 则使用 CodeFormer 管道、sigma [0,15]。本项目保留双阶段覆盖，只适度收窄幅度范围；这是针对强噪声预览的自定义折中，并非这些项目的官方预设。显式配置原始范围仍能恢复官方幅度，没有全局增益或网格衰减。

两阶段默认混合真实 JPEG/WebP：第一阶段 85%/15%，第二阶段 80%/20%。JPEG quality [30,95]、显式 4:2:0，WebP quality [45,98]。JPEG2000 仍可通过 compression.probabilities 选择，默认 0%；重压缩默认关闭。HYPIR 使用浮点 DiffJPEG，两者存在量化和色度重建差异。实测其补零至 16 倍数会影响非整倍数 LQ 的边缘，不移植这一行为，不承诺逐像素复刻。

训练和标定共享 degrade_pair 返回 LQ 与监督 GT；degrade_image 仅用于返回 LQ 的诊断。native 输出在 LQ 网格量化、跳过末端 grain，放大它不能替代训练输出。独立算子随机流确保 grain 开关不影响其他抽样，trace 记录实际尺寸、核、采集噪声、编码和 grain。

沿用配对 50% 水平翻转、裁剪后退化；HYPIR 默认也启用水平翻转且关闭旋转。其 crop_type=none 要求预先准备 512 图片，本项目保留多尺寸裁剪及小图反射填充。逐样本独立 resize 抽样无需 HYPIR 弥补批内共享参数的训练队列。

旧 stage.noise.grain_probability/grain_sigma/grain_correlation 字段已删除，迁移到 final_grain。监督策略进入 resume identity，需要重新标定，不能静默沿用旧策略。本次不改变训练损失，不声称已获得模型质量提升。


## 编辑、参考图、掩码和图层

| 字段 | 填什么 | 模型范围 |
| --- | --- | --- |
| reference_sources | 有序参考图目录列表 | FLUX Kontext/Fill、FLUX.2/3、支持编辑/参考的 Qwen |
| mask_source | 与目标主体匹配的掩码目录 | FLUX Fill |
| layer_sources | 独立 RGBA 前景图层目录列表 | Qwen layered |
| control_source | 对齐控制目录 | Wan |

这些目录是目标的条件，不是独立训练组。列表顺序决定参考/图层顺序。按[模型指南](models.md#编辑参考图与图层)选择对应变体；不支持的组合在读权重前报错。

## 视频数据

```toml
[dataset.groups.main]
kind = "video"
source = "/data/videos"
cache_directory = "/data/cache/wan"
resolution = [512, 512]
batch_size = 1

[dataset.groups.main.video]
target_frames = [81]
extraction = "head"
max_frames = 129
target_fps = 16.0
fps_resampling = "fractional"
```

这个窗口配置供 Wan 使用：帧数满足 1+4k。head 从开头提取；需要不同窗口策略时查看 target_frames、extraction、stride、samples 的[参数表](parameters.md)。

H3 复制其专用示例：24 FPS、timestamps 重采样、17n+5 帧、batch_size=1，并准备原生 AV JSONL 记录。不要将普通视频目录配置直接改 family 当作 H3 输入。

## 哪些变化需要重建缓存

| 变化 | 操作 |
| --- | --- |
| 图片、txt、prompt、参考图/掩码/图层 | 重新 cache |
| VAE、文本编码器、tokenizer、resolution 或桶规则 | 重新 cache |
| 仅学习率、训练长度、输出周期 | 不因这些字段本身要求重新编码；续训仍检查数值配置身份 |
| 修复训练 | 缓存后继续保留 LQ/HQ，训练会实时读取 |

每次 cache 完整重建，全部成功后替换旧缓存。不要用同时运行的训练读取正在重建的缓存。

---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
