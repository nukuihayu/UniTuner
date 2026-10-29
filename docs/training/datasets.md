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

合成退化先从 HQ 原图裁出训练尺寸，再构造同尺寸 LQ；不把整张原图缩成训练切片。HQ 保持不变。1536 切片示例：

```toml
[dataset.groups.main]
resolution = [1536, 1536]
enable_bucket = false

[dataset.groups.main.degradation]
enabled = true
scale_range = [1.0, 8.0]
noise_strength = [0.85, 0.85]
processing_deadzone = 0.1
smear_strength = [0.0, 0.8]
grain_strength = [0.0, 0.8]
grain_scale_range = [1.0, 4.0]
sharpen_strength = [0.0, 0.4]
```

倍率与退化强度独立，在 `scale_range` 的对数空间内均匀连续采样，支持 1～16，包括 2.3、5.7 等小数；默认 [1,8] 时，[1,2]、[2,4]、[4,8] 的概率相同；每个倍率区间都覆盖弱／中／强约 20%／40%／40% 的连续强度，不是三种退化配方。也可显式填写 `scales=[2.5,4,6,8]` 等概率抽取指定倍率，填写后优先于 scale_range。实际低清网格按 `floor(H/倍率)`、`floor(W/倍率)` 取整，因此实际倍率会有像素取整误差。1536 的 4×、6×、8× 分别得到 384、256、192 网格，最后恢复至 1536。

不再选择“常规／颗粒／涂抹”类别。所有样本共用同一条成图流程。对从零开始的强度范围，使用 `g(z)=max(0,(z−t)/(1−t))`，其中 `t=processing_deadzone=0.1`。这让低强度的一段概率质量落在精确零上，零点以上仍连续变化；不是追加互斥的风格开关。t=0 恢复纯连续采样，t=1 将所有从零开始的处理强度归零。正的下界表示明确要求操作持续开启，不应用死区；固定范围保持固定。

降噪程度 u、颗粒保留程度 v、全局噪声程度 n 均服从 Beta(1.5,2)。平滑范围的位置使用 g(u)；颗粒保留使用连续的 `v(1−0.5u)`，不再应用第二个死区。默认 `noise_strength=[0.85,0.85]`，将采集噪声标准差整体收至 Real-ESRGAN 范围的 85%，同时保留其高噪声尾部；两阶段同时启用时仍按方差 70%／30% 分配，只有一个阶段启用时获得完整的全局增益。独立 grain 的强度和启用概率均乘 `p=max(0.15,1−0.85s)`，与共享采集强度 s 反向耦合；因此会产生明确的无 grain 样本。第二阶段 RGB 噪声将色度缩至 0.35 后重新归一 RMS，减少紫绿斑点而不改变该阶段分配到的噪声能量。Gaussian/Poisson 各占 50%，低照度第一阶段仍强制使用线性光 shot/read 噪声。

管道按算子边界分成独立层：resize 只改尺寸；blur 统一 Gaussian／广义 Gaussian／平台核，不生成运动或失焦 PSF；compression 只编解码；noise 合并阶段噪声与残余颗粒；smear、sinc、sharpen 各自独立。阶段内 resize／blur／noise 随机换序。smear、sharpen、grain 先随机排列；末端将“回目标尺寸 + sinc”作为不可拆分操作，只允许 `resize+sinc → compression` 或 `compression → resize+sinc`。第二阶段噪声每个样本只执行一次：默认 75% 留在阶段内接受后续交付处理，25% 延迟至整个交付组之后，在最终 HQ 网格形成细噪点。

将 `noise_strength=[0,0]` 可关闭所有额外噪声，将 `smear_strength=[0,0]` 可关闭内容平滑与相机降噪；blur 配置仍独立生效。若要恢复带精确零点的连续噪声增益，可设置 `noise_strength=[0,1]`；`processing_deadzone=0` 会移除零起点范围的死区。

| 来源场景 | 默认比例 | 相关处理 |
| --- | --- | --- |
| 普通照片 | 25% | 光学模糊、缩放、采集噪声、交付编码 |
| 网络转载 | 35% | 中间压缩、缩放、较强交付压缩 |
| 低照度 | 40% | 线性光域 shot/read 噪声、保边降噪和色度平滑 |

来源场景仍决定采集噪声和历史编码倾向，但不会选择独占的成图分支。共享强度 s∈[0,1] 使用平滑累积分布 `F(s)=0.1s+1.8s²−0.9s³`，密度处处为正，覆盖整个范围；按强度三等分仅作统计时，弱／中／强概率为 20%／40%／40%。通过一次均匀抽样反解此分布，不选择类别、不切换流程。s 关联模糊、采集噪声与编码质量，倍率另行独立抽样。这是采样强度比例，不是逐图视觉损伤的精确比例；图像内容、算子开关和独立增益仍影响结果。指定固定倍率或 scales 列表会覆盖倍率分布；降噪与残留颗粒另有连续变化，不会所有损伤始终一起变强。

第一阶段的 resize、blur、noise 随机排列，网络转载 compression 固定在这些处理之后；第二阶段对 resize、blur、noise 全排列。交付处理层先随机排列，随后执行受约束的末端编码组；组内严格二选一：`resize+sinc → compression` 或 `compression → resize+sinc`。前一种概率为 `0.55+0.30s`，强采集噪声更倾向在放大后压缩，避免低清压缩块被继续放大。禁用 sinc 或 compression 时保留其余操作，但 resize 仍负责回目标尺寸。

关联物理参数的 80% 抽样仍围绕场景强度，20% 独立探索配置的完整范围，覆盖弱模糊配强压缩等组合和区间尾部；这不意味着所有组合等概率。处理增益仍遵循原有分布，倍率独立按对数均匀分布抽样。

`grain_scale_range=[1,4]` 按对数均匀分布改变噪声生成网格相对输出的缩小倍率；提前在低清网格加入颗粒时，最细只能达到该网格的单像素，放大后自然形成粗颗粒。只放大噪声残差并加入图像，避免为了改变颗粒尺寸而再次缩放图像结构。噪声包含细粒亮度和更宽的相关色度分量，颗粒强度连续控制残差幅度。

所有空间核使用分段分辨率系数：`r<1024` 时 `k(r)=clamp((r/1024)^(1/4),0.5,2)`，`r>=1024` 时使用 `clamp(sqrt(r/1024),0.5,2)`；`r` 是该层执行时图像的实际短边。64、256、512、768、1024、1280、2048、4096 分别约为 0.5、0.707、0.841、0.931、1.0、1.118、1.414、2.0。blur 以 Real-ESRGAN x4 为强度基准：1×～4× 保持原始 sigma 与覆盖率，6×／8× 的显式 sigma 分别乘 2/3／1/2，第二阶段 blur 概率也从 80% 动态降到约 53%／40%，避免把更强的缩放损失重复计入模糊。blur 在 resize 后执行时还会乘 `当前短边/目标短边`，按真实执行网格补偿后续放大。第一阶段 sigma 基准为 [0.2,3.0]，第二阶段为 [0.2,1.5]。sinc 支撑与截止频率继续按实际分辨率调整；smear、grain 与 sharpen 按实际缩小网格约束，其中 sharpen 强度使用与噪声相同的温和平方根网格增益，平滑强度另按实际降采样倍率 d 乘 `min(1,2/d)`。

阶段采集噪声按添加时的实际网格使用 `sqrt(clamp(当前短边/目标短边,1/16,1))` 缩放标准差。若随机顺序为 resize 后再加噪声，4×／6×／8× 网格的噪声增益约为 0.50／0.41／0.35，避免放大后形成过强粗斑；resize 前在目标网格添加的噪声保持 1.0。延迟到完整交付组之后的 HQ 细噪声同样保持 1.0。

`first.noise.exposure=[0.75,1]` 在线性光域降低采集曝光，加入噪声后恢复；未裁断时方差为 `signal*shot/exposure + read/exposure²`。第二阶段 exposure=[1,1]。这仍是已处理 RGB 上的近似，不是 RAW/demosaicing 模拟。双线性和双三次降采样带抗混叠，统一使用显式像素坐标规则；不会引入几何美颜或人脸形变操作。

编码使用真实 JPEG/WebP/JPEG2000 往返，默认启用 JPEG/WebP。JPEG 质量范围两阶段均为 [30,95]；各阶段 `recompression_probability=0.25` 可在移动压缩块原点后重编码，再裁回原图位置。压缩层不再隐式触发模糊。


sinc 核使用 Kaiser 窗（beta=4）渐弱有限支撑的边界，保留轻微振铃并减少短核归一化放大的人工描边。支撑大小随 `k(r)` 放大，截止频率按 `1/k(r)` 缩小并限制到 π。一个 1024 基准的 7×7 核、截止频率 1.889 的阶跃测试中，过冲从约 24.3% 降至 3.3%；这不是对所有图像的统一误差保证。

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
