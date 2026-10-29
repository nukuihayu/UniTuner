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

合成退化采用 Real-ESRGAN 固定双阶段，HQ 监督图不变：

1. 对退化源执行可选 USM；第一阶段 blur/sinc → resize → noise → compression。
2. 第二阶段可选 blur/sinc → resize → noise。
3. 等概率执行「回最终 LQ 尺寸 + final sinc → compression」或「compression → 回最终 LQ 尺寸 + final sinc」。
4. 在真实 LQ 网格量化到 8-bit，然后单独用 bicubic 放大到 HQ 尺寸，满足同尺寸 restoration 训练接口。对齐放大后不再增加噪声、压缩或滤波。

```toml
[dataset.groups.synthetic.degradation]
enabled = true
scales = [2.0, 4.0, 6.0, 8.0]
reference_resolution = 400
resolution_exponent = 0.5
second_order_probability = 1.0
use_usm = true
resize_antialias = false

[dataset.groups.synthetic.degradation.first.noise]
grain_probability = 0.05
[dataset.groups.synthetic.degradation.second.noise]
grain_probability = 0.05
```

官方参数本身没有按分辨率动态适配。官方数据集先预裁剪至 400，再在退化后裁 256 的训练块；400 是这里选择的参考尺寸，并不是官方相机标定值。本项目只在样本开始计算一次空间系数：`k=clamp((HQ短边/reference_resolution)^resolution_exponent,0.5,2)`。默认 256/400/512/1024/1280/2048 对应约 0.80/1.00/1.13/1.60/1.79/2.00；两个阶段的 sigma、7～21 基础核支撑、sinc 支撑和截止频率使用这个系数，不按中间网格反复缩放。设 `resolution_exponent=0` 关闭分辨率扩展。噪声强度、编码质量、算子概率不会因分辨率改变。

基础采样区间与官方一致：blur sigma 为 [0.2,3]/[0.2,1.5]；第二阶段 blur 概率 80%；阶段 sinc 各占启用滤波的 10%，与普通 blur 互斥；final sinc 概率 80%。阶段 sinc 截止频率根据基础核尺寸使用 π/3～π 或 π/5～π，final 使用 π/3～π，不加 Kaiser 窗。基础 resize 范围为 [0.15,1.5]/[0.3,1.2]。

默认开启本项目扩展 `high_scale_protection=true`，专门防止将 x4 预设直接用于 x6/x8 时过度缩小。令 `r=min(1,4/倍率)`，仅对第二阶段向下 resize 的抽样值应用 `s'=1-(1-s)*r`，x4/x6/x8 的下限分别是 0.30/0.533/0.65。final sinc 空间系数改为 `max(0.5,k*r)`，避免高倍率对齐放大后振铃和低通范围过大。第一阶段、第二阶段 blur、噪声幅度和压缩质量不变；x1～x4 不受此开关影响。设 false 可恢复未保护的官方 resize 范围。各算子使用独立随机流，切换噪声类别不会改变后续几何或编码参数。

noise 每次只选一种：默认 grain 5%，其余 95% 在 Gaussian/Poisson 中等概率抽取。两阶段 Gaussian sigma 为 [1,30]/[1,25]，Poisson scale 为 [0.05,3]/[0.05,2.5]，灰噪声概率 40%。Poisson 恢复 BasicSR 的量化颜色数定义。grain 是本项目扩展的轻量相关亮度噪声，sigma [1,8]/255；不作为第三次噪声叠加。每个双阶段样本至少抽到一次 grain 的概率为 9.75%。

压缩每阶段默认执行一次：JPEG/WebP 第一阶段 85%/15%，第二阶段 80%/20%；JPEG2000 支持但默认 0%。JPEG quality [30,95]，WebP [45,98]；重压缩默认关闭。使用真实 OpenCV 编解码，和官方 GPU DiffJPEG 不保证逐像素一致。禁用两阶段 grain、将压缩概率设为 [1,0,0]，同时设 `resolution_exponent=0`，可检查官方固定参数基线。

完整固定参数基线还需要 `high_scale_protection=false`，不能把默认的高倍率保护称为官方算法。

现代方法核查与采用范围：

- [HYPIR 2025 配置](https://github.com/XPixelGroup/HYPIR/blob/main/configs/sd2_train.yaml) 继续使用标准两阶段范围、0.1/0.1/0.8 sinc 和 resize_back，支持先生成 LQ 再对齐的实现选择。

- [ResShift 配置](https://github.com/zsyOAOA/ResShift/blob/master/configs/realsr_swinunet_realesrgan256.yaml) 保留 Real-ESRGAN 参数并增加 second_order_prob=0.5、use_sharp=false。本项目提供 `second_order_probability` 和 `use_usm` 选项，默认仍为 1.0/true。
- [BSRGAN](https://github.com/cszn/BSRGAN) 扩展摄像噪声和随机排列；为保持官方顺序，本版本没有引入 shuffle/ISP。
- [DiffBIR](https://github.com/XPixelGroup/DiffBIR)、[SUPIR](https://github.com/Fanghua-Yu/SUPIR) 表明现代恢复训练仍需选择适合目标数据的退化分布；SUPIR 区分轻退化模型。未据此臆造新噪声参数或移植生成式退化网络。
- 可显式开启 `resize_antialias` 研究抗混叠 resize，默认 false 与官方保持一致。WebP 和低概率 grain 是本次明确限定的扩展，不声称已经获得训练指标提升。

旧的 scenario_probabilities、noise_strength、方差分摊、late_noise_probability、独立 grain/smear/sharpen 等字段已删除，会明确报错；请迁移示例并重新校准训练。策略版本进入 resume identity，不能静默续用旧退化分布。诊断 API 支持 `trace=[]`，记录实际执行尺寸、核、noise 类别/幅度和 codec/quality。`output_mode="native"` 仅供直接预览调用；数据集及标定要求 aligned。

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
