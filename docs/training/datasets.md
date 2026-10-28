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

合成退化保持 HQ 的构图、亮度和颜色作为监督目标，重点模拟低分辨率照片的细节损失与处理痕迹。默认不添加雾、雨、划痕或曝光变化。输入最好来自清晰、低噪声、压缩较轻的原始照片；已有损伤会与合成退化叠加。

| 场景 | 默认占比 | 关联的退化 |
| --- | --- | --- |
| 普通照片 | 55% | 轻度光学模糊、降采样、弱噪声、可选最终压缩 |
| 网络转载 | 30% | 中间编码、再次缩放、较强最终压缩、较高概率的轻微锐化 |
| 低照度照片 | 15% | 线性光域信号相关 shot/read 噪声、保边降噪、色度平滑，随后降采样与编码；不改变目标曝光 |

默认从 `scales = [2, 3, 4]` 等概率选择倍率，各约占 33.33%；填写 `[2, 3, 4, 8]` 时各占 25%，不额外偏向最大倍率。共享退化强度使用 Beta(2,3) 分布，以中轻度为主并保留较强样本；模糊、噪声、压缩围绕同一强度小幅波动。倍率选择不决定强度。场景比例是可调整的训练先验，不是对真实照片来源比例的测量。

```toml
[dataset.groups.main.degradation]
enabled = true
scales = [2, 3, 4]
# 普通照片 / 网络转载 / 低照度；固定网络转载可设 [0.0, 1.0, 0.0]。
scenario_probabilities = [0.25, 0.35, 0.40]
processing_probabilities = [0.30, 0.45, 0.25]
```

采集阶段保持光学模糊在前，噪声可在首次缩放前或后添加，各占一半；低照度噪声后立即关联降噪。只有网络转载可能在此编码一次。交付阶段随机排列缩放、局部模糊和噪声操作；若触发第二阶段噪声，一半在该阶段执行，一半推迟到放大回 HQ 尺寸后执行，同一样本不会因此重复添加第二阶段噪声。仍需放大时，最终编码等概率安排在放大前或后；若颗粒路径已提前放大至 HQ，则在 HQ 网格上编码。这覆盖放大后的细颗粒和不同尺度的压缩痕迹。每阶段还可按下述概率额外重编码。JPEG/WebP 为默认编码，JPEG2000 需显式选择。

第一阶段 `blur.sigma = [0.4, 2.8]` 使用 1024 短边基准，实际乘以 `clamp(sqrt(min(H,W)/1024), 0.5, 1.5)`；1536 方图约为 `[0.490, 3.429]`。这是有限增长的训练先验，不是镜头物理定律。第二阶段 `[0.15, 0.65]` 直接使用操作所在网格的像素单位，不随 HQ 放大。居中的运动核与圆形失焦核保持配对对齐。sinc 截止频率使用本地像素网格；最终 sinc 概率为 0.1，核尺寸为 7–21。

bilinear/bicubic 降采样启用抗混叠，默认放大使用 bilinear/bicubic。所有缩放共用同一坐标实现；使用 `scale_factor` 时设置 `recompute_scale_factor=True`，让坐标按取整后的实际尺寸计算，bilinear/bicubic 明确使用 `align_corners=False`，减少连续缩放的系统性位置漂移。该设置不消除滤波误差或任意输入配对错位。轻微锐化发生在当前交付网格，概率为普通/低照度 0.3、转载 0.6，强度为 0.1–0.4。噪声与编码范围的详细单位见[参数参考](parameters.md#degradationconfig)。低照度处理是已处理 RGB 图像上的近似，不包含完整 RAW 去马赛克流程；真实成对样本仍应直接使用真实 LQ，不再叠加合成退化。更改退化分布后重新校准修复时间步，合成退化的数据集身份也会变化；旧退化任务不能作为完全一致的断点续训，新任务可沿用文本缓存并从已有权重开始训练。

默认处理混合为常规 30%、颗粒 45%、涂抹 25%。颗粒在交付半尺寸或全尺寸生成，混合细粒 shot/read 残差和空间相关颗粒；`grain_strength=[0.75,1.5]` 控制幅度（允许 0～4）。涂抹采用两次保边滤波与色度平滑，`smear_strength=[0.5,1.0]` 控制混合（允许 0～1）。颗粒分支中还有 `grain_smear_probability=0.45` 的概率先在低分辨率上涂抹、再加颗粒，覆盖细节被擦除但仍有噪声的成图。

每个已触发的压缩阶段默认 `recompression_probability=0.25`，额外真实编码／解码一次，错开块边界并裁回，不移动场景几何。末次编码随机在放大前或后执行。第一／第二阶段 JPEG 默认质量为 [45,95]／[35,95]。来源场景为普通／转载／低照度 [0.25,0.35,0.40]，严重程度使用 Beta(2,2)。这些是更强的已处理 RGB 训练先验，不是校准过的 RAW 流程，也不是对参考人像的定量拟合。

启用模拟退化后两类风格都默认参与，HQ 不变。TOML 显式值优先于新默认值；改变退化分布后重新校准 sigma。退化策略版本纳入续训标识，防止静默按旧轨迹继续。全尺寸滤波和重复编码增加 CPU 开销。设 processing 为 [1,0,0] 跳过颗粒／涂抹，两阶段 compression 中 recompression_probability=0 可跳过额外编码。


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


### 按局部内容调整手机成图退化

更新后的相机噪声在线性光域采样 `first.noise.exposure=[0.75,1.0]`，加入 shot/read 噪声后恢复曝光。未发生裁断时，恢复后的方差为 `signal*shot/exposure + read/exposure²`：增加曝光不足带来的噪声，而非刻意把目标整体变暗。第二阶段 exposure 默认为 [1,1]，设置 [1,1] 可关闭曝光恢复。这仍是已处理 RGB 上的近似，不是 RAW 传感器标定。

彩色噪声使用比亮度颗粒更宽的相关范围；颗粒增益默认收敛到 [0.75,1.5]，减少前段传感器噪声与后段颗粒叠加过重的情况。`smear_detail_protection=0.6`（允许 0～1）根据局部对比度连续减弱强边缘附近的涂抹，低对比微纹理仍会被去除；设置 0 恢复均匀混合。不引入人脸语义掩码、几何美颜、水印或周期纹理。

使用清晰 DIV2K 人像及另一场景，各以 8 个固定种子验证默认流程。这验证了输出和分布覆盖，不代表对参考照片的定量拟合，也未证明恢复训练质量提升。跨策略版本相同 seed 不保证各随机操作逐项一致。请用完整 PNG 或原像素局部图观察颗粒，缩略图会隐藏噪声。数值策略版本升级为 4，应新开训练并重新校准。


### 先压缩、后模糊

`compression_blur_probability=0.25` 在 LQ 网格上增加一次历史编码，再立即进行光学滤波，随后才进入手机成图处理和末次交付编码。复用 `second.compression` 与 `second.blur` 配置，使已有压缩块、振铃也能被模糊，而非只有最后新生成的压缩痕迹。设为 0 关闭此路径；显式将第二阶段 blur_probability 或 compression_probability 设为 0 也会关闭它。否则使用本路径独立概率，不再乘以普通第二阶段的两个概率。

默认常规／颗粒／单独涂抹比例改为 [0.30,0.45,0.25]。颗粒分支中 45% 先涂抹，涂抹加颗粒的期望总比例从 12.25% 提升到 20.25%，单独涂抹从 40% 降至 25%。这些是处理分支概率，不是互斥的完整退化配方，其他阶段仍可能叠加模糊、噪声和压缩。沿用局部对比度保护，不加入人脸关键点处理或几何变换；不能保证每张低分辨率样本的五官观感都改善。
