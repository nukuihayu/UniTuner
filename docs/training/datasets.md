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

以 [HYPIR 官方配置与实现](https://github.com/XPixelGroup/HYPIR/tree/b61d107c6cef38f01a93c7833558869731cfa8c1/HYPIR/dataset) 为基础，对齐 OMGSR 较轻的噪声、二阶段 blur 和 JPEG 分布，使用下述受限缩放，扩展不同尺寸/倍率、多种压缩和末端 10% grain。

1. 裁剪/配对水平翻转后，默认 USM 关闭，使用原始 HQ 作为退化源和监督 GT。显式 `use_usm=true` 时启用 HYPIR USM（固定核 51、weight=0.5、threshold=10）；`sharpen_target=true` 同时锐化监督 GT，设 false 仅锐化退化源。
2. 第一阶段 blur/sinc → resize → Gaussian/Poisson → 压缩。
3. 第二阶段可选 blur/sinc → resize → Gaussian/Poisson。
4. 等概率选择原生 LQ [resize+sinc] → 压缩或压缩 → [resize+sinc]。
5. bicubic 回 HQ，10% 概率只添加一次亮度 grain，最后裁剪和 8-bit 量化。

```toml
[dataset.groups.synthetic.degradation]
enabled = true
scale_range = [1.0, 8.0]
# scales = [2.0, 4.0, 6.0, 8.0]
reference_resolution = 512
resolution_exponent = 0.5
use_usm = false
sharpen_target = true

[dataset.groups.synthetic.degradation.first.compression]
probabilities = [0.85, 0.10, 0.05]
jpeg_backend = "diffjpeg"

[dataset.groups.synthetic.degradation.second.compression]
probabilities = [0.8, 0.15, 0.05]
jpeg_backend = "diffjpeg"

[dataset.groups.synthetic.degradation.final_grain]
probability = 0.1
sigma = [1.0, 8.0]
correlation = [0.3, 0.8]
```

HYPIR 基准为 512/x4，原始两阶段 resize 范围为 [0.15,1.5] / [0.3,1.2]。当前默认采用 [OMGSR 的较缓和范围](https://github.com/wuer5/OMGSR/blob/8996613b95ffeba8653a214f2ca0c04abe10b74a/dataset/params_realesrgan.yml)：[0.3,1.5] / [0.6,1.2]；升/降/保持概率仍为 [0.2,0.7,0.1] / [0.3,0.4,0.3]。额外硬约束：每次中间 resize 的高宽都不得小于最终 `floor(HQ/倍率)`，即使显式配置旧范围也生效。1280、4× 时中间图最小 320，不会先缩到 111 再放回 320。HYPIR 和 OMGSR 原版均允许这种中途过度缩小；这里是有意的行为差异。总倍率 `scale_range` 在 [1,16] 内按对数均匀分布连续抽样；显式 `scales` 优先。

空间适配是扩展：HQ 系数 `k=clamp((HQ短边/reference_resolution)^resolution_exponent,0.5,2)`。执行网格缩小时，混合核 sigma/支撑和 sinc 逆截止频率采用 `min(k,max(1,clamp((执行短边/reference_resolution)^resolution_exponent,0.5,2)))`，避免将 HQ 的额外核放大照搬到小 LQ 网格；低分辨率 HQ 仍保留温和适配。512 时系数为 1；指数设 0 可保留官方固定像素核。USM 固定 51，不缩放噪声幅度或 JPEG 块。极小 LQ 网格无法使用 Torch reflect 时采用重复反射。

两阶段 Gaussian/Poisson 各 50%、灰噪声 40%；sigma [1,15]/[1,12]、Poisson scale [0.05,2]/[0.05,1]，对齐 OMGSR，直接用于 resize 后网格，不加增益或衰减。加噪后 clip、不立即量化；仅 Poisson 采样信号预先量化并统计唯一值数。

混合 blur 六类概率 [0.45,0.25,0.12,0.03,0.12,0.03]，基础支撑 7～21、sigma [0.2,3]/[0.2,1.5]。第二阶段 blur 50%，每个已启用阶段有 10% 替换 sinc，final sinc 80%。阶段 cutoff 为 π/3～π（基础核<13）或 π/5～π，final 为 π/3～π，不加窗。

压缩概率顺序为 JPEG/WebP/JPEG2000：第一阶段 [0.85,0.10,0.05]，第二阶段 [0.8,0.15,0.05]。JPEG 默认使用 HYPIR 同算法浮点 DiffJPEG，浮点 quality [60,95]/[60,99]、4:2:0、官方量化表、最近邻色度重建，保留补零至 16 倍数的边界行为。已用固定 HYPIR 版本生成的 12 组参考样本验证该编码器的数值一致性。`jpeg_backend="opencv"` 可切换真实 JPEG。WebP quality [60,98]，JPEG2000 [250,900]/[200,900]；真实编码器先量化到 8-bit，quality 取整。DiffJPEG quality 必须严格介于 0 和 100。默认关闭额外重压缩。JPEG 下限对齐 OMGSR 的 60，第二阶段上限以 99 代替官方 100，避开 DiffJPEG 的除零奇点；WebP 下限同步提高到 60，JPEG2000 保留原区间。

对照官方纯 JPEG 编码器时，将两阶段压缩概率设为 [1,0,0]，末端 grain 概率设为 0；中间尺寸保护仍生效，因此不等同于原版完整管道。同 seed 不保证整图逐像素相同：本项目局部 NumPy 随机流不同于官方全局 Python/NumPy/Torch 随机流。逐图独立抽样代替批内共享参数的训练队列。不修改损失，也不声称已验证模型质量提升。

末端 grain RMS [1,8]/255、相关半径 [0.3,0.8] 个输出像素，只在回原尺寸后抽取一次，native 输出跳过。训练/标定共享 `degrade_pair` 返回 LQ 和监督 GT；`degrade_image` 仅用于诊断。trace 记录实际网格、核、噪声和编码。旧网格/高倍率开关及阶段 grain 字段仍拒绝；监督及分布变化需要重新标定，并影响 resume identity。


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
