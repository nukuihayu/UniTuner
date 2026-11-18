# 文本编码精度与量化

简体中文 · [English](../en/training/text-encoding.md) · [训练指南](README.md)

文本编码器可以读取普通浮点或 Kitchen 打包权重，也可以在 conditioning 缓存或推理前逐层量化。它与 denoiser 的精度独立：`model.text` 控制文本编码，`execution.quantization` 控制训练基模。

<details>
<summary>本页导航</summary>

- [本地单文件权重](#本地单文件权重)
- [Tokenizer 来源](#tokenizer-来源)
- [缓存前逐层量化](#缓存前逐层量化)
- [读取提前转换的编码器](#读取提前转换的编码器)
- [效果与重新缓存](#效果与重新缓存)

</details>

## 本地单文件权重

所有模型的 `model.text_encoder` 均可指向本地单文件权重；使用 safetensors 分片时可指定任一标准编号分片，所有分片必须齐全。原有目录形式继续支持。FLUX 的底层 T5/CLIP 加载器均支持单文件；CLI 的辅助 CLIP 仍沿用 `model.assets/text_encoder/` 目录约定。按实际键名识别已支持的家族格式，仍严格检查键名、形状与量化布局，不转换任意 checkpoint。

```toml
[model]
family = "z-image"
variant = "turbo"
text_encoder = "/root/autodl-tmp/models/z-image-turbo/text_encoders/qwen_3_4b.safetensors"
```

单文件的模型配置按以下顺序解析：同名 `.json`（例如 `qwen_3_4b.json`）、同目录 `config.json`、内置发布配置。内置配置覆盖 Z-Image 的 Qwen3-4B、FLUX 的 T5-XXL/CLIP-L、FLUX.2 Dev/Klein、FLUX3/Krea2 的 Qwen3-VL-4B、Qwen Image 和 Qwen Image 2.1。自定义规格应提供对应 JSON；形状和键名不匹配会严格报错。

MOSS 的 Qwen3.5 单文件需要同名 JSON 或同目录 `config.json`；Wan 使用已有原生配置；MiniMaxH3 继续通过 `model.minimax_h3.text_config` 指定配置目录。模型配置不会在加载时联网下载。Tokenizer/processor 仍独立解析；未配置默认 Hub 来源的模型，单文件会以其父目录作为本地 tokenizer 候选，也可以显式指定 `model.text.tokenizer.source`。缓存身份包含全部分片和实际使用的模型 JSON，配置或权重变化后需重新缓存。

## Tokenizer 来源

Tokenizer／processor 默认从下表的 Hub 仓库获取，仅下载词表、模板及处理器配置，不下载编码器权重，也不执行远程代码。原生文本编码器仍从 `model.text_encoder` 读取本地权重和模型配置。

| 模型 | 默认仓库／子目录 |
| --- | --- |
| Z-Image | `Tongyi-MAI/Z-Image` / `tokenizer` |
| FLUX | T5：`google/t5-v1_1-xxl`；CLIP：`openai/clip-vit-large-patch14` |
| FLUX.2 Dev | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` |
| FLUX.2 Klein 4B／9B | `Qwen/Qwen3-4B`／`Qwen/Qwen3-8B` |
| Qwen Image | `Qwen/Qwen-Image` / `tokenizer`；processor：`Qwen/Qwen-Image-Edit` / `processor` |
| Krea2 | `Qwen/Qwen3-VL-4B-Instruct` |
| Wan | `Wan-AI/Wan2.1-T2V-14B` / `google/umt5-xxl` |
| MiniMaxH3 | `MiniMaxAI/MiniMax-H3` / `processor` |

```toml
[model.text.tokenizer]
# source = "D:/models/my-tokenizer" # 本地目录，或其他 Hub repo ID
# subfolder = "tokenizer"          # 显式 source 默认读取根目录
revision = "main"                  # 推荐固定为 commit SHA
offline = false
```

省略 `source` 使用上表默认值；省略 `revision` 使用仓库默认分支。`offline = true` 只读取已有 Hub 缓存，本地目录始终不会联网。受限仓库沿用 Hugging Face 登录凭据。可通过 `--set model.text.tokenizer.offline=true` 等 CLI 覆盖。完全离线时也可以设置 `HF_HUB_OFFLINE=1`。

FLUX 可用同样字段的 `[model.text.clip_tokenizer]` 独立指定 CLIP 来源；Qwen Image 可用 `[model.text.processor]` 独立指定图像处理器来源。未指定这些表时沿用主 tokenizer 的离线开关；revision 分别设置。Qwen Image 显式指定主 tokenizer 的 `source` 且省略 processor 表时，使用同一来源，目录需包含完整处理器资产。其他多模态模型直接在 tokenizer 表选择完整 processor 目录。

MOSS Image、FLUX3、Qwen Image 2.1 暂保留本地文本目录作为默认来源，也支持上述显式覆盖。Wan 通过 tokenizer 表选择词表；MiniMaxH3 的 `text_processor` 可省略。缓存身份包含实际词表、模板、processor 内容，切换来源或 revision 后需重新运行 cache。

## 缓存前逐层量化

安装量化依赖，在原训练配置中添加：

```bash
uv sync --locked
```

```toml
[model.text]
precision = "bf16"
quantization = "float8_e4m3fn"
```

`model.text_encoder` 继续指向本地源资产，然后运行 `unituner cache train.toml --device cuda`。量化可以降低 GPU 权重内存，但准备阶段仍需足够主机内存容纳浮点源模型，源文件不会被修改。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `precision` | 省略 | 浮点计算 dtype；可选 `fp32`、`fp16`、`bf16`，省略沿用调用方默认值，命令行缓存和生成通常为 FP32 |
| `quantization` | `none` | 浮点源的目标权重布局；`none` 保留已量化文件的布局 |
| `quantization_rotation_size` | `256` | ConvRot 组大小，至少为 4 的 4 的幂，须满足层尺寸要求 |
| `quantization_group_size` | `16` | 分组 4／6 位布局的组大小，须满足布局与层尺寸要求 |
| `max_length` | 家族默认 | 仅在原生接口支持时改变 token 预算；量化不改变各家族的限制 |

量化支持 `float8_e4m3fn`、`float8_e5m2`、`mxfp8`、`nvfp4`、`int8_tensorwise`、`int8_rowwise`、`int8_convrot`、`convrot_w4a4`、`convrot_w4a8`、`asym_w4a8_int8` 和 `w6a8_int8`。`fp8` 与 `convrot-int8` 分别是 E4M3 和 INT8 ConvRot 的别名。AWQ／SVDQuant 需要校准，当前文本运行时不接受它们。

所有布局只量化权重，输入保持浮点，线性层临时反量化后做浮点乘法。格式名中的 A4／A8 不会启用激活量化。FP8／FP4 存储不代表使用低位矩阵乘法加速，当前层仍需临时浮点显存。尺寸不兼容会报错；需要选择性保留不兼容层时，先独立转换并使用 `--on-incompatible keep`。

这些设置同样用于训练预览与独立生成的文本加载；具体文本设备沿用对应流程。Denoiser、VAE、文本编码器可使用不同精度，降低文本精度不会降低 VAE 精度。只希望改变文本浮点计算时，设置 `precision` 并保留 `quantization = "none"`。

## 读取提前转换的编码器

为输出目录准备对应的模型 config；tokenizer、chat template 和 processor 可使用上述独立来源，然后转换权重：

```bash
uv run unituner weights convert models/qwen3/model.safetensors models/qwen3-fp8/model.safetensors \
  --family qwen3 --precision float8_e4m3fn --device cuda
```

将 `model.text_encoder` 改为 `models/qwen3-fp8`，保留 `model.text.quantization = "none"`，即可使用文件已有量化格式。显式指定的目标格式必须与已有量化层匹配。设置 `model.text_source_format` 时，需与源格式一致。

文件转换也支持 `--precision bf16`、`fp16` 和 `fp32`。转回浮点不能恢复量化时丢失的信息。比较在线与离线转换时，请保持精度、量化参数和设备一致。

| 模型家族 | 文本编码器转换预设 |
| --- | --- |
| Z-Image、FLUX.2 Klein | `qwen3` |
| MOSS Image | `qwen3-5` |
| FLUX | 主文本目录 `t5`，辅助 CLIP 目录 `clip`，分别转换 |
| FLUX.2 Dev | `mistral` |
| FLUX3、Qwen Image 2.1、Krea2、MiniMaxH3 | `qwen3-vl` |
| Qwen Image | `qwen2-5-vl` |
| Wan | `umt5`，保留原生拓扑元数据及独立 tokenizer 资产 |

FLUX 的在线设置同时应用于 T5 和 CLIP；需要不同布局时分别离线转换。保留原有辅助资产目录结构。转换分片输入时会合并输出，输出目录不应混留旧分片或过期索引。

MiniMaxH3 实际只使用语言层 0–49，在线量化自动保护第 0 和 49 层。离线转换完整、未截断的 Qwen3-VL 时，另加 `--exclude '*layers.49.*'`；更推荐转换匹配其原生截断拓扑的文件。文本加载器接受首末层等路径已经量化的文件，并保留其打包存储，不要求重新转换。键、形状、布局及显式量化参数仍须通过校验。

## 效果与重新缓存

敏感文本层会自动保留较高精度。请使用与模型匹配的编码器、分词器和处理器资产。

更换编码器、分词器、revision、精度或量化设置后，重新运行 `unituner cache train.toml --device cuda`。比较训练效果时请固定这些设置。将特征保存为 FP32 不能恢复低精度编码时丢失的信息。

---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
