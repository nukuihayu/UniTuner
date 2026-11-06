# 模型与变体指南

简体中文 · [English](../en/training/models.md)

[文档目录](../README.md) · [训练指南](README.md) · [示例配置](../examples/README.md)

先按任务选择模型，再准备对应的权重、VAE 和文本编码器。下表列出的流匹配训练支持 LoRA、LoHa、LoKr 和全参数训练；修复训练按变体选择。

## 从任务选择模型

| 想完成的任务 | 选择与准备 |
| --- | --- |
| 主体或风格微调 | 选择已有完整本地模型文件的图像模型，准备图片和同名文本 |
| 图像编辑 | 选择编辑变体，准备目标图、原图和编辑指令 |
| 图层训练 | 使用 Qwen layered，准备合成图与独立 RGBA 前景图层 |
| 一步图像修复 | 选择支持修复的模型，准备已对齐的 LQ/HQ 或合成退化配置 |
| 视频或音视频 | 使用 Wan 或 MiniMaxH3，按对应示例准备帧数、帧率和音频 |

切换模型时，同时核对 `family`、`variant`、`version`、VAE 和文本编码器；只修改模型名称无法替换整套模型文件。

## 模型选择

| `model.family` | `model.version` | `model.variant` | 数据与条件 | 修复训练 | 流匹配示例 |
| --- | --- | --- | --- | --- | --- |
| `z-image` | 不设置 | `base`／`turbo` | 图片和文本 | 是 | [Z-Image](../examples/flow.toml) |
| `flux` | 不设置 | `dev`／`kontext`／`fill` | Kontext／Fill 需要一张条件图 | 仅 dev | [FLUX](../examples/flux-flow.toml)、[Fill](../examples/flux-fill-flow.toml) |
| `flux2` | `dev`／`klein-4b`／`klein-9b` | `base`／`distilled`；dev 仅 distilled | 文本和可选有序参考图 | 是 | [FLUX.2](../examples/flux2-flow.toml)、[参考图](../examples/flux2-reference-flow.toml) |
| `flux3` | 不设置 | `base`／`distilled` | 文本，最多 10 个参考来源 | 是 | [FLUX3](../examples/flux3-flow.toml) |
| `moss-image` | 不设置 | `base`／`sft`／`edit` | edit 需要参考图和编辑指令 | Base/SFT | [MOSS](../examples/moss-flow.toml)、[编辑](../examples/moss-edit-flow.toml) |
| `qwen-image` | 不设置 | `original`／`edit`／`edit-2509`／`edit-2511`／`layered` | 文本、编辑参考图或前景图层 | 仅 original | [原版](../examples/qwen-image-flow.toml)、[编辑](../examples/qwen-image-edit-flow.toml)、[图层](../examples/qwen-image-layered-flow.toml) |
| `qwen-image-2-1` | 不设置 | `base` | 文本和可选参考图 | 是 | [Qwen2.1](../examples/qwen-image-2-1-flow.toml) |
| `krea2` | 不设置 | `raw`／`turbo` | 图片和文本 | 是 | [Krea2](../examples/krea2-flow.toml) |
| `bluepixel` | 不设置 | `turbo` | 一张参考图和编辑指令；尺寸为 64 的倍数 | 是 | [BluePixel](../examples/bluepixel-flow.toml) |
| `wan` | `2.1`／`2.2` | `t2v`／`i2v` | 图片或视频；i2v 需要条件帧 | 否 | [Wan](../examples/wan-flow.toml) |
| `minimax-h3` | 不设置 | `t2va`／`fl2va`／`ref2va` | 图片或音视频记录 | 否 | [H3](../examples/minimax-h3-train.toml) |

仅 FLUX.2 和 Wan 需要设置 `model.version`。Base、Turbo 和蒸馏变体应使用各自的权重与采样参数。

## VAE 加载

将匹配的 VAE 文件或目录填入 `model.vae`：

| 模型 | 需要准备的 VAE |
| --- | --- |
| Z-Image、FLUX | 对应模型的 16 通道 VAE |
| FLUX.2、FLUX3、MOSS | 配套的 FLUX.2 32 通道 VAE |
| Qwen Image、Krea2 | 对应模型的 16 通道 VAE；Qwen layered 使用支持 RGBA 的版本 |
| Qwen Image 2.1 | 配套的 64 通道 RGBA VAE |
| BluePixel-Turbo | 模型快照中的 `vae/` 目录，包含 DCAE-F64 配置和权重 |
| Wan | 对应的 16 通道 VAE；可使用受支持的 `.pth`、`.pt` 或 safetensors 文件 |
| MiniMaxH3 | 配套的视频 VAE 和音频 VAE，按 H3 示例分别填写 |

标准模型通常可直接使用其配套文件。加载报错时先检查是否拿错模型版本或组件；需要转换文件时参见[权重转换](../weight-conversion.md)。

## 公共本地资产

| 字段 | 填写内容 |
| --- | --- |
| `model.weights` | 主模型文件、分片索引或受支持的模型目录 |
| `model.vae` | 配套 VAE 文件或目录 |
| `model.text_encoder` | 配套文本编码器文件或目录 |
| `model.assets` | 辅助文件根目录；训练配置中省略时使用主模型路径的父目录 |

FLUX 的 `model.text_encoder` 指向 T5，CLIP 放在 `model.assets/text_encoder/`；Wan 2.1 I2V 的图像编码器放在 `image_encoder/model.safetensors`。BluePixel 的 `assets` 指向包含 `tokenizer/` 的模型快照根目录。

模型权重需要自行下载。分词器和处理器可以使用默认来源，也可指定本地目录、版本和离线模式，见[文本编码](text-encoding.md)。文本编码器格式通常可自动识别，只有需要明确限定时才设置 `model.text_source_format`。

## 编辑、参考图与图层

参考目录按文件主体与目标匹配：

```toml
[dataset.groups.main]
source = "/data/targets"
cache_directory = "/data/cache"
reference_sources = ["/data/references"]
```

`targets/a.png` 对应 `references/a.png`，同名 `a.txt` 写编辑指令。多个目录的顺序就是参考图顺序。

| 模型 | 条件要求 |
| --- | --- |
| FLUX Dev | 不使用参考图 |
| FLUX Kontext／Fill | 恰好一个参考目录；Fill 可另设 `mask_source` |
| FLUX.2 | 可选一个或多个有序参考目录 |
| FLUX3 | 最多 10 个参考目录 |
| BluePixel | 流匹配训练恰好一个参考目录；修复训练使用 LQ/HQ 配对 |
| MOSS edit | 至少一个参考目录 |
| Qwen edit | 最初的 edit 需要参考图；2509／2511 可不提供 |
| Qwen original／layered | 不使用编辑参考目录；layered 使用 `layer_sources` |
| Qwen2.1 | 可选有序参考目录，可独立设置参考图面积预算 |

目标和参考图按各自长宽比缩放、中心裁剪。原始尺寸相同且已对齐的图片保持配对关系；不同构图不会自动配准。修改图片、参考来源或裁剪设置后，重新运行 `cache`。

Qwen2.1 可将目标固定为方图，同时保留参考图长宽比：

```toml
[dataset.groups.main]
source = "/data/targets"
cache_directory = "/data/cache"
resolution = [1024, 1024]
enable_bucket = false
reference_sources = ["/data/references"]
reference_resolution = [1024, 1024]
```

`reference_resolution` 表示参考图的面积预算，不会强制输出为方图；该选项仅用于 Qwen2.1。需要对齐原图和区域遮罩时，保持原始尺寸与坐标一致，并省略这一覆盖项。参考图中的遮罩不会自动变成损失遮罩。

FLUX Fill 的 `mask_source` 是重绘遮罩，目标、原图和遮罩应同尺寸且对齐。

## 必要的模型专属参数

| 配置 | 何时使用 |
| --- | --- |
| `method.guidance` | FLUX／FLUX.2 Dev 的训练条件；与推理 CFG 不同 |
| `method.qwen_image.remove_base` | Qwen layered 是否排除合成图监督 |
| `dataset.groups.main.wan.single_frame` | Wan I2V 从图片组选择条件帧和目标帧 |
| `model.minimax_h3` | H3 视频／音频编码器及处理器文件 |
| `dataset.groups.main.minimax_h3` | H3 缓存、音频和时长设置 |
| `method.minimax_h3` | H3 视频／音频噪声与损失设置 |

完整字段见[参数参考](parameters.md)，后验和时间校准设置见[修复训练](restoration.md)。

## 视频与音视频

Wan 的目标帧数和最大帧数需要满足 `1 + 4k`。2.1 I2V 需要配套图像编码器，2.2 I2V 按对应示例填写。先运行 [Wan 缓存](../examples/wan-cache.toml)，再使用 [Wan 训练](../examples/wan-flow.toml)。

H3 要求 batch 为 1、分辨率为 32 的倍数，视频为 24 FPS，目标帧数满足 `17n + 5`，最大帧数至少为 5；音频为 32 kHz 立体声。先运行 [H3 缓存](../examples/minimax-h3-cache.toml)，再使用 [H3 训练](../examples/minimax-h3-train.toml)。

数据目录、条件帧和生成方法见[视频与音视频指南](video.md)。

## MOSS-Image-Edit

选择 `family = "moss-image"`、`variant = "edit"`，使用匹配的编辑权重或适配器。仅修改 variant 不会让普通底模学会编辑。

1. 按[编辑训练示例](../examples/moss-edit-flow.toml)准备目标图、同名指令和参考目录。
2. 执行 `unituner cache CONFIG --device cuda`，再执行 `unituner train CONFIG --device cuda`。
3. 按[编辑生成示例](../examples/moss-edit-generate.toml)填写 `sampling.reference_images`。
4. 在对应 ComfyUI 节点中选择 **Moss-Image-Edit**，并使用匹配的适配器文件。

编辑配置使用单独缓存，不能混用 Base/SFT 缓存。多参考图会增加显存占用。MOSS edit 暂不支持 TREAD、一步修复和蒙版背景锁定。

## 预览默认参数

省略 `steps` 和 `guidance` 时使用对应变体的默认值：

| 模型 | Steps | Guidance |
| --- | --- | --- |
| Qwen Image 2.1 | 25 | 1 |
| Qwen Image / edit / layered | 25 | 4 |
| Z-Image base / turbo | 50 / 9 | 4 / 0 |
| FLUX Dev / Kontext / Fill | 25 | 3.5 / 2.5 / 30 |
| FLUX.2 Dev / Klein base | 50 | 4 |
| FLUX.2 Klein distilled | 4 | 1 |
| BluePixel-Turbo | 4 | 3 |
| FLUX3 base / distilled | 50 / 16 | 4 / 不使用 CFG |
| MOSS Image | 50 | 3 |
| Krea2 raw / turbo | 28 / 8 | 5.5 / 1 |

---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
