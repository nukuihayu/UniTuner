# 模型与变体指南

简体中文 · [English](../en/training/models.md)

[文档目录](../README.md) · [训练指南](README.md) · [示例配置](../examples/README.md)

当前十个家族共用[训练方法](README.md)，各自维护原生网络、VAE、文本编码和条件装配。
下表中的流匹配均有 adapter（LoRA／LoHa／LoKr）与 full 入口；默认按基模精度自动选择，
流匹配与八个图像家族的修复任务均可显式选择 CUDA BF16 AMP。十个家族的 adapter 训练均可使用[训练前权重量化](quantized-base.md)或原生量化基模；full 训练要求浮点权重。

<details>
<summary>本页导航</summary>

- [从任务选择模型](#从任务选择模型)
- [VAE 加载](#vae-加载)
- [模型选择](#模型选择)
- [公共本地资产](#公共本地资产)
- [编辑、参考图与图层](#编辑参考图与图层)
- [必要的模型专属参数](#必要的模型专属参数)
- [视频与音视频](#视频与音视频)
- [MOSS-Image-Edit](#moss-image-edit)

</details>

## 从任务选择模型

主体或风格微调可以先使用已有完整本地资产的图像模型；图像编辑需要支持编辑的变体与对应参考条件；Qwen layered 使用独立前景图层；视频和音视频任务则需要额外遵守帧数、时间采样与音频规则。

复制配置前，先根据 checkpoint 确定准确的家族、必要的 version 与 variant，再检查 VAE、文本编码器、辅助 bundle 和数据条件。仅修改 `model.family` 不能完成模型迁移。

## VAE 加载

标准 VAE 直接填写 `model.vae` 路径，无需先运行 `weights import` 或提供拓扑 JSON。缺少原生元数据时使用对应家族的发布配置；原生元数据优先保留自定义结构，显式配置与元数据冲突时仍报错。

| 家族 | 默认结构与外部格式 |
| --- | --- |
| Z-Image | 标准 16 通道 VAE；自动识别 Diffusers／ComfyUI 键 |
| FLUX | 标准 16 通道 VAE；自动识别原生／Diffusers 键 |
| FLUX.2、FLUX3、MOSS | 共用标准 32 通道 packed VAE；自动识别原生／Diffusers 键 |
| Qwen Image、Krea2 | 标准 16 通道因果 VAE；自动识别官方／ComfyUI 键；Qwen layered 由变体选择 RGBA |
| Qwen Image 2.1 | 独立 64 通道 RGBA VAE；使用本项目发布结构默认值和原生键 |
| Wan 2.1／2.2 | 共用 16 通道因果 VAE；支持官方 `.pth`／`.pt` state dict 和 safetensors，自动识别 ComfyUI 键 |
| MiniMax-H3 | 内置视频／音频 VAE；自动移除视频的 `first_stage_model.`／`video_vae.`／`vae.` 和音频的 `audio_vae.`／`vae.` 前缀 |

加载仍严格检查完整键集合和张量形状，不跳过未知权重。非标准结构可通过 Python 加载器的 `config` 或 `weights import --topology` 显式提供。

## 模型选择

| `model.family` | `model.version` | `model.variant` | 数据与条件 | 修复训练 | 流匹配示例 |
| --- | --- | --- | --- | --- | --- |
| `z-image` | 不设置 | `base`／`turbo` | 图片＋Qwen3 文本 | 是 | [flow](../examples/flow.toml) |
| `flux` | 不设置 | `dev`／`kontext`／`fill` | T5＋CLIP；Kontext／Fill 一张条件图 | 仅 dev | [FLUX](../examples/flux-flow.toml)、[Fill](../examples/flux-fill-flow.toml) |
| `flux2` | `dev`／`klein-4b`／`klein-9b` | `base`／`distilled`；dev 仅 distilled | 原生文本、可选有序参考图 | 是 | [FLUX.2](../examples/flux2-flow.toml)、[参考图](../examples/flux2-reference-flow.toml) |
| `flux3` | 不设置 | `base`／`distilled` | Qwen3-VL、最多10个参考来源 | 是 | [FLUX3](../examples/flux3-flow.toml) |
| `moss-image` | 不设置 | `base`／`sft`／`edit` | Qwen3.5 文本、packed VAE | Base/SFT | [MOSS](../examples/moss-flow.toml) |
| `qwen-image` | 不设置 | `original`／`edit`／`edit-2509`／`edit-2511`／`layered` | Qwen2.5-VL，编辑参考图或独立前景 | 仅 original | [原版](../examples/qwen-image-flow.toml)、[编辑](../examples/qwen-image-edit-flow.toml)、[图层](../examples/qwen-image-layered-flow.toml) |
| `qwen-image-2-1` | 不设置 | `base` | Qwen3-VL、可选参考图，独立64通道vae | 是 | [Qwen2.1](../examples/qwen-image-2-1-flow.toml) |
| `krea2` | 不设置 | `raw`／`turbo` | Qwen3-VL 逐层文本特征，原生固定选层 | 是 | [Krea2](../examples/krea2-flow.toml) |
| `wan` | `2.1`／`2.2` | `t2v`／`i2v` | 图片／视频，原生控制与条件帧 | 否 | [Wan](../examples/wan-flow.toml) |
| `minimax-h3` | 不设置 | `t2va`／`fl2va`／`ref2va` | 图片／音视频，原生 JSONL AV 记录 | 否 | [H3](../examples/minimax-h3-train.toml) |

不能给其他家族添加 `model.version`。蒸馏／Turbo 不构成独立训练方法，也不意味着可以混用
base 权重、文本编码器或采样时间表。变体必须与权重内拓扑一致，schema 校验不会读取权重验证这一点。

## 公共本地资产

- `model.weights`：主模型 safetensors、编号分片、索引 JSON 或 Diffusers 目录；支持对应家族的 ComfyUI／Diffusers 权重，训练必填。
- `model.vae`：匹配模型的本地 VAE 权重；缓存、预览、修复／生成需要。
- `model.text_encoder`：本地文本编码器目录或原生组件权重，形式由家族决定。
- `model.assets`：辅助 bundle 根目录；省略时为 denoiser 文件的父目录。

辅助 bundle 固定使用 `text_encoder/`（FLUX 的 CLIP）和
`image_encoder/model.safetensors`（Wan2.1 I2V 的 CLIP）。编码器权重不会自动下载。
Tokenizer／processor 使用独立的默认 Hub 来源，也可覆盖本地目录、revision 和离线模式，见[文本编码配置](text-encoding.md#tokenizer-来源)。
FLUX 的 `model.text_encoder` 仍是 T5，不是 CLIP。Wan2.2 I2V 不使用2.1的独立图像编码器。
H3 需要独立的 AV VAE／处理器，见下面的专属参数。

`[model.text] max_length` 是公共可选项，但只在支持覆盖的家族开放：Z-Image 默认512，
MOSS4096（上限4096），FLUX512（上限512），Krea2 512。其他家族使用固定原生文本预算，拒绝显式覆盖。
Krea2 使用编码器原生特征选层。

Qwen、Qwen2.1、Krea2、H3 默认按实际权重键名自动识别格式，无需填写 `model.text_source_format`。兼容原生 base、Transformers，以及语言键为 `model.layers.*`、视觉键为 `visual.*` 或 `model.visual.*` 的 Qwen/Comfy 混合布局。Comfy-Org 的 `qwen3vl_8b_bf16.safetensors` 可直接加载到规格匹配的 Qwen2.1 编码器。
需要约束输入时仍可显式指定 `"transformers"` 或 `"qwen"`；`null`/`"auto"` 使用自动识别。识别读取所有分片的键名，不根据文件名或配置中的 architecture 猜测格式。混合语言前缀、重复视觉别名、缺失/多余键、形状不符或非有限数值仍会报错。H3 保持自身宽度与截断层要求，并校验完整模型中最终丢弃的层。
FLUX.2/FLUX.3、MOSS、Z-Image 同样按已支持的语言键名前缀识别；FLUX T5/CLIP 保留既有别名校验，Wan 自动识别统一 `encoder.*` 前缀。VAE 保留各家族的格式/拓扑校验，Z-Image 也支持自动移除统一 `vae.*` 前缀。

## 编辑、参考图与图层

在数据组中填写条件来源：

```toml
[dataset.groups.main]
source = "/data/targets"
cache_directory = "/data/cache"
reference_sources = ["/data/reference-a", "/data/reference-b"]
```

按目标文件主体匹配每个条件来源，有序列表决定参考顺序；条件图不是额外的独立训练样本。
只使用一个参考来源时删除第二项。参考图处理规则：先根据目标图
选择 bucket，再将每张参考图 resize、中心裁剪到同一个 bucket。不提供额外的参考图缩放策略。
原始尺寸相同且已经对齐的图片对使用相同缩放比例和裁剪坐标；不同构图的参考图不会因此自动配准。
放大使用 Lanczos，缩小使用 area，缩放尺寸按 half-up 舍入。

| 模型 | 条件约束 |
| --- | --- |
| FLUX Dev | 不接受 reference_sources |
| FLUX Kontext／Fill | 恰好一个来源；Fill 可另设 mask_source |
| FLUX.2 | 有序参考来源，与目标 bucket 使用相同几何预处理 |
| FLUX3 | 最多10个来源，与目标 bucket 使用相同几何预处理 |
| Qwen edit 系列 | 最初 edit 必须有参考，2509／2511 可无参考 |
| Qwen2.1 | 有序参考来源，与目标 bucket 使用相同几何预处理 |
| Qwen original／layered | 不接受编辑参考目录；layered 使用 layer_sources |

训练缓存与生成推理是独立的预处理边界；本节描述训练数据集。改变预处理后需要重新生成缓存。

## 必要的模型专属参数

| 配置位置 | 用途／适用范围 |
| --- | --- |
| `method.guidance` | FLUX／FLUX.2 Dev 的 learned guidance embedding；不是推理 CFG；修复入口不开放此项 |
| `method.qwen_image.remove_base` | Qwen layered 是否排除合成图监督，默认 false |
| `dataset.groups.main.wan.single_frame` | Wan I2V 的独立条件／目标帧索引，仅 image 数据；与公共 control_source 互斥 |
| `model.minimax_h3` | H3 audio_vae、text_config，及可选 text_processor／topology／text_device |
| `dataset.groups.main.minimax_h3` | H3 cache_seed、音频解码／修复边界、实验时长策略 |
| `method.minimax_h3` | H3 sigma／focus 区间、video_shift=12、audio_shift=3、条件清洁度与 audio_loss_weight=1 |

`dataset.groups.main.control_source` 已是公共字段，当前能力检查仅允许 Wan；共享名字不代表其他家族已实现。
流匹配使用各模型的缓存后验策略；
修复训练的公共 `method.restoration.posterior` 可选值及默认策略见[修复指南](restoration.md)。

## 视频与音视频

Wan 的 `dataset.groups.main.video.target_frames` 和 max_frames 必须满足1+4k，原生视频 VAE 保留因果时间压缩。
使用 [wan-cache](../examples/wan-cache.toml) 生成缓存，训练配置选择同一缓存和模型。
Wan 显式 FP32 模式关闭 autocast，注意力仍保留原生 BF16 Q/K/V，不能理解为所有内部运算都 FP32；
显式 CUDA BF16 AMP 遵循公共执行策略。
2.1 I2V 需要独立图像编码器；2.2 I2V 沿用其原生条件路径。完整 FLF2V 应用不在当前支持范围。

H3 当前 batch_size=1、分辨率32对齐；video 使用24FPS timestamp 重采样，目标帧数17*n+5，
max_frames 至少5，原生音频32kHz立体声。JSONL 是其 AV 记录格式，不是图像修复配对格式。
先用 [H3 cache](../examples/minimax-h3-cache.toml)，再用 [H3 train](../examples/minimax-h3-train.toml)；
已有显式拆分缓存可用 `caches import` 和 [导入说明JSON](../examples/minimax-h3-cache-import.json)。
当前缓存生产尚不支持 teacher-caption 记录，不应仅因 schema 可读取就用于训练。

视频预览与生成分别使用各家族原生条件；[示例索引](../examples/README.md) 提供具体入口。

[填写数据集](datasets.md) · [完整参数默认值](parameters.md)

## MOSS-Image-Edit

`model.family="moss-image"`、`model.variant="edit"` 复用 MOSS Base/SFT 权重，启用
[ComfyUI PR #17296](https://github.com/Comfy-Org/ComfyUI/pull/17296) 的 **Moss-Image-Edit** 格式
（`index_timestep_zero_text_start`）。它需要编辑训练或匹配的编辑 adapter；修改 variant 本身不会赋予底模编辑能力。
此格式不等同于 PR 中 ai-toolkit 的两种格式：文本编码器只读取指令，参考图只通过 packed VAE latent 输入。

参考 token 使用零时间，只读取同组参考；文本和目标使用当前时间并读取全部 token。
参考图编号从 1 开始，frame/text 坐标固定为 0，目标坐标仍位于有效文本末尾。
训练只对目标加噪并计算 flow loss，参考分支保留梯度；支持全参数和已有 adapter 算法。

- [训练示例](../examples/moss-edit-flow.toml)：`source` 放编辑后的目标图及同名指令 `.txt`，
  `reference_sources` 放按文件名 stem 匹配的原图，可按顺序提供多个目录。训练参考图使用目标 bucket 尺寸。
- 先执行 `unitunner cache <配置> --device cuda`，再执行 `unitunner train <配置> --device cuda`。
  编辑缓存保存有序参考 latent，不能混用原版 MOSS 缓存；缓存身份包含参考图内容。
- [生成示例](../examples/moss-edit-generate.toml)：使用 `sampling.reference_images`；参考图保留各自尺寸，
  按 16 像素对齐，不采用 FLUX.2 的参考图缩放策略。预览使用同样字段。
- 推理从目标纯噪声开始。默认缓存逐层参考 K/V，在一次采样内供各步及相同跳层路径的 CFG 共用；
  SLG／跳层分支分别构建缓存，采样结束即释放，不跨训练更新复用。
  edit 默认不跳过无条件分支的第 22 层；Base/SFT 原有默认值保持不变。
- 导出 adapter 沿用 ComfyUI 的 `diffusion_model.*` 键。在该 PR 的节点中选择 **Moss-Image-Edit**，
  不要选择 ai-toolkit。外部仓库的自定义 adapter 文件需先转换为项目支持的发布键与缩放格式。

当前 edit 不支持 TREAD 或 restoration 方法，也不包含蒙版背景锁定。参考 K/V 缓存以显存换计算，
多参考图和不同跳层路径会增加缓存占用。

训练继续使用原生 MOSS 的显式精度转换和残差计算，因此不承诺与 ComfyUI 融合内核逐位一致。

---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
