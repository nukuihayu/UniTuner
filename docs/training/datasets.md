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

### 合成超分退化

合成退化先从 HQ 原图裁出训练尺寸，再构造同尺寸 LQ；不把整张原图缩成训练切片。HQ 保持不变。1536 切片示例：

```toml
[dataset.groups.main]
resolution = [1536, 1536]
enable_bucket = false

[dataset.groups.main.degradation]
enabled = true
scale_range = [1.0, 8.0]
smear_strength = [0.0, 0.8]
grain_strength = [0.0, 1.5]
grain_scale_range = [1.0, 4.0]
sharpen_strength = [0.0, 0.4]
```

倍率在 `scale_range` 内按对数均匀分布连续采样，支持 1～16，包括 2.3、5.7 等小数；[1,8] 在 [1,2]、[2,4]、[4,8] 三个区间的概率相同。也可显式填写 `scales=[2.5,4,6,8]` 等概率抽取指定倍率，填写后优先于 scale_range。实际低清网格按 `floor(H/倍率)`、`floor(W/倍率)` 取整，因此实际倍率会有像素取整误差。1536 的 4×、6×、8× 分别得到 384、256、192 网格，最后恢复至 1536。

不再选择“常规／颗粒／涂抹”类别。所有样本共用同一条成图流程，连续采样平滑、残余颗粒、颗粒尺寸和锐化强度。平滑程度 u 与独立的颗粒保留程度 v 都服从 Beta(1.5,2)；平滑范围内的位置为 u，颗粒范围内的位置为 v(1−0.5u)，模拟降噪更强时通常残留更少颗粒，同时保留平滑与噪声并存的样本。锐化在配置范围内均匀采样。它们是合成分布先验，并非对真实设备的标定。将某个强度范围设为 [0,0] 可完全关闭对应操作；关闭颗粒不等于关闭 first/second 中的噪声。

| 来源场景 | 默认比例 | 相关处理 |
| --- | --- | --- |
| 普通照片 | 25% | 光学模糊、缩放、较弱采集噪声、交付编码 |
| 网络转载 | 35% | 中间压缩、缩放、较强交付压缩 |
| 低照度 | 40% | 线性光域 shot/read 噪声、保边降噪和色度平滑 |

来源场景仍决定采集噪声和历史编码倾向，但不会选择独占的成图分支。共享损伤程度服从 Beta(2,2)，模糊、噪声和编码质量围绕它变化；降噪与残留颗粒另有连续变化，不会所有损伤始终一起变强。

采集阶段首先施加光学模糊，再随机安排噪声和缩放；低照度噪声后带相机降噪，网络场景可有中间编码。第二阶段随机安排缩放、局部模糊和噪声；一半被选中的第二阶段噪声延后到最终放大后生成，不重复添加。随后统一到低清网格，按概率执行先压缩后模糊及 sinc，进行连续强度的局部内容保护平滑和锐化。最后编码可在放大前或放大后执行；残余颗粒在恢复 HQ 尺寸后添加，若选择放大前编码，颗粒表示交付后处理残差。

`grain_scale_range=[1,4]` 按对数均匀分布改变噪声生成网格相对输出的缩小倍率。只放大噪声残差并加入图像，避免为了改变颗粒尺寸而再次缩放图像结构。噪声包含细粒亮度和更宽的相关色度分量，颗粒强度连续控制残差幅度。

第一阶段模糊 sigma 默认 [0.4,2.8]，按 `clamp(sqrt(min(H,W)/1024),0.5,1.5)` 缩放；1536 方图约为 [0.490,3.429]。第二阶段 sigma=[0.15,0.65] 使用当前网格像素。平滑默认单次温和保边滤波，`smear_detail_protection=0.6` 用局部对比度保护轮廓。实际降采样倍率 d 会把额外平滑强度乘以 `min(1,2/d)`，避免强缩小与强涂抹叠加。4×、6×、8× 分别保留 1/2、1/3、1/4 的平滑强度。该限制不恢复降采样已丢失的细节。

`first.noise.exposure=[0.75,1]` 在线性光域降低采集曝光，加入噪声后恢复；未裁断时方差为 `signal*shot/exposure + read/exposure²`。第二阶段 exposure=[1,1]。这仍是已处理 RGB 上的近似，不是 RAW/demosaicing 模拟。双线性和双三次降采样带抗混叠，统一使用显式像素坐标规则；不会引入几何美颜或人脸形变操作。

编码使用真实 JPEG/WebP/JPEG2000 往返，默认启用 JPEG/WebP。JPEG 质量范围第一阶段 [40,95]、第二阶段 [30,95]；各阶段 `recompression_probability=0.25` 可在移动压缩块原点后重编码，再裁回原图位置。`compression_blur_probability=0.25` 在低清网格编码后模糊已有块效应和振铃，复用第二阶段配置；设为 0 或显式关闭第二阶段 blur/compression 可禁用。其概率独立于普通第二阶段两个开关的非零值。

旧配置请删除 `processing_probabilities` 和 `grain_smear_probability`，改用连续强度范围；这些旧键会明确报错，不会静默忽略。修改分布后重新校准 restoration timestep/sigma，并新开训练；策略版本参与 resume identity。真实配对 LQ 不需要合成退化。样图只验证生成过程和视觉覆盖，不能据此声称恢复训练质量或人脸保真度提升。

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
