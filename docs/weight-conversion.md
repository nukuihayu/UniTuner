# 权重精度转换

简体中文 · [English](en/weight-conversion.md) · [文档首页](README.md)

`unituner weights convert` 将 safetensors 权重转换为浮点或支持的量化格式，可用于减少权重存储、准备可复用的量化基模，或导出供其他工具使用的浮点权重。

默认输出使用 ComfyUI 主模型格式。需要保留输入格式时使用 `--output-format source`，尤其是为同一应用准备权重时。请按目标加载器选择格式。

<details>
<summary>本页导航</summary>

- [开始转换](#开始转换)
- [支持的精度](#支持的精度)
- [按模型保护敏感层](#按模型保护敏感层)
- [选择需要转换的层](#选择需要转换的层)
- [导入 AWQ 与 SVDQuant](#导入-awq-与-svdquant)
- [文件、磁盘空间与结果](#文件磁盘空间与结果)
- [运行时兼容性](#运行时兼容性)

</details>

## 开始转换

在项目根目录同步默认依赖：

```bash
uv sync --locked
```

将模型转换为 FP8：

```bash
uv run unituner weights convert models/model.safetensors models/model-fp8.safetensors \
  --precision float8_e4m3fn --device cuda
```

转换为 MXFP8，或将已有量化权重反量化为 BF16：

```bash
uv run unituner weights convert models/model.safetensors models/model-mxfp8.safetensors \
  --precision mxfp8 --device cuda
uv run unituner weights convert models/model-mxfp8.safetensors models/model-bf16.safetensors \
  --precision bf16 --device cuda
```

输入支持单文件、任意编号分片、safetensors 索引 JSON、Diffusers 的 `transformer/` 目录或包含该目录的模型根目录。输出为一个 safetensors 文件，放入 ComfyUI 的 `models/diffusion_models/` 后使用对应模型加载节点。量化文件使用对应 Kitchen 格式；普通 BF16 无需 Kitchen。输入与输出必须使用不同路径。

```bash
uv run unituner weights convert models/z_image_turbo_bf16.safetensors models/z_image_turbo_fp8.safetensors \
  --family z-image --precision float8_e4m3fn --device cuda
```

训练可直接把 ComfyUI 单文件或 Diffusers 目录填入 `model.weights`，无需先执行 `weights import`。发布模型使用默认结构，本地 `config.json` 可提供结构参数；`model.variant` 和 `model.version` 仍用于选择训练变体。支持 Z-Image、FLUX、FLUX.2、FLUX3、MOSS、Qwen Image、Qwen Image 2.1、Krea2、Wan 和 MiniMaxH3 各自发布的布局。

Qwen Image 2.1 支持发布的 ComfyUI 文件和索引分片。自定义模型结构需要匹配的 `config.json`。量化文件载入后可能比磁盘文件占用更多内存，请预留计算空间。

`weights import --kind transformer` 也默认导出 ComfyUI 布局；自定义结构可提供 `--topology`，需要原生布局时指定 `--output-format native`。

## 支持的精度

| `--precision` | 转换方式 |
| --- | --- |
| `fp32`、`fp16`、`bf16` | 浮点转换；量化输入先反量化 |
| `float8_e4m3fn`、`float8_e5m2` | 张量级缩放 FP8 |
| `mxfp8` | 每 32 元素 E8M0 缩放 |
| `nvfp4` | 每 16 元素 FP8 缩放，加 FP32 全局缩放 |
| `int8_tensorwise` | 张量级 INT8 |
| `int8_rowwise` | 按输出行缩放 INT8 |
| `int8_convrot` | 按行缩放，附带 ConvRot |
| `convrot_w4a4`、`convrot_w4a8` | 4 位权重，4 位或 8 位激活布局 |
| `asym_w4a8_int8` | 分组 4 位码本权重 |
| `w6a8_int8` | 分组 6 位均匀权重 |
| `awq_w4a16` | 导入已离线校准的权重 |
| `svdquant_w4a4` | 导入已离线校准的权重及低秩分支 |
| `keep` | 校验并重新封装，保留量化数据 |

W/A 描述部署时的权重与激活精度。文件转换保存权重和布局参数；激活量化由消费该文件的推理运行时执行。

## 按模型保护敏感层

默认 `--family auto` 优先读取原生拓扑元数据。没有元数据时，根据家族特有的键识别；键结构不明确时显式指定 `--family`。家族名称与训练一致：`z-image`、`flux`、`flux2`、`flux3`、`moss-image`、`qwen-image`、`qwen-image-2-1`、`krea2`、`wan`、`minimax-h3`。

```bash
uv run unituner weights convert models/qwen21.safetensors models/qwen21-int8.safetensors \
  --family qwen-image-2-1 --precision int8_convrot --device cuda --on-incompatible keep
```

输入输出、时间和条件投影、归一化、调制等敏感层会自动保留浮点精度。转换结果中的 `skipped` 会列出受保护或尺寸不兼容的层。

`--include` 只能缩小范围，不能覆盖自动保护。不在预设范围内的模型可使用 `--family generic`，使用前请检查转换效果。

### 文本编码器保护

使用 `--family qwen2-5-vl|qwen3|qwen3-vl|qwen3-5|mistral|t5|umt5|clip` 选择对应预设，自动保留敏感层、词嵌入和多模态组件的精度。

```bash
uv run unituner weights convert models/text_encoder/model.safetensors models/text_encoder-fp8.safetensors \
  --family qwen3-vl --precision float8_e4m3fn --device cuda
```

保留转换后的模型配置、分词器和处理器资产，或单独配置它们的来源。文本缓存与生成设置见[文本编码精度与量化](training/text-encoding.md)。

## 选择需要转换的层

在家族允许的范围内，量化处理非空二维 `*.weight`（以及根键 `weight`）；偏置和其他张量保持原样。这个规则不能判断矩阵来自线性层还是嵌入层，应按实际模型的加载器选择需要量化的层：

```bash
uv run unituner weights convert models/model.safetensors models/model-int8.safetensors \
  --precision int8_convrot --device cuda --rotation-size 256 \
  --include 'transformer_blocks.*.weight' --exclude '*norm*' '*embed*' \
  --on-incompatible keep
```

匹配使用完整参数键的 glob，并区分大小写；排除规则优先。请给模式加引号，避免 shell 提前展开。浮点转换对选中的所有浮点张量生效，整数缓冲区保持不变；没有布局描述的整数权重会报错，以免将未知打包数据误当成普通权重。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--family` | `auto` | 家族保护策略；含 `generic`、支持的基模及文本编码器 |
| `--device` | `cpu` | 转换使用的 `cpu` 或 `cuda` 设备 |
| `--include` / `--exclude` | 无 | 一个或多个参数键 glob |
| `--rotation-size` | `256` | ConvRot 的旋转组大小，必须为至少 4 的 4 的幂 |
| `--group-size` | `16` | 分组 4/6 位布局的量化组大小，至少为 4，且整除 16 或为 16 的倍数 |
| `--on-incompatible` | `error` | 不满足布局形状要求时终止；`keep` 保留该层浮点权重，并在结果中列出原因 |
| `--backend` | 自动 | 可选 `eager`、`cuda`、`triton`；后两者需满足 Kitchen 的硬件与内核条件 |
| `--layout-spec` | 无 | 为规范打包的离线权重补充布局描述 |

CPU 转换可用 `--backend eager`，其他后端需要兼容的硬件和精度。`--on-incompatible keep` 可跳过尺寸不兼容的层，但不能跳过缺失校准或损坏文件等错误。

## 导入 AWQ 与 SVDQuant

这两个布局的上游 `quantize()` 明确要求离线校准，不能仅凭浮点权重生成正确结果。先使用相应校准流程得到 **Kitchen 规范打包数据**，再补充布局描述。以 AWQ 为例，输入需包含：

```text
transformer_blocks.0.attn.to_q.weight
transformer_blocks.0.attn.to_q.weight_scale
transformer_blocks.0.attn.to_q.weight_zeros
```

将以下内容保存为 `calibrated-layout.json`。键是模块前缀，不含 `.weight`；根权重使用空字符串。尺寸与 dtype 必须匹配实际校准产物：

```json
{
  "transformer_blocks.0.attn.to_q": {
    "format": "awq_w4a16",
    "orig_shape": [3072, 3072],
    "orig_dtype": "bf16",
    "group_size": 64
  }
}
```

```bash
uv run unituner weights convert models/calibrated.safetensors models/model-awq.safetensors \
  --precision keep --layout-spec calibrated-layout.json
```

AWQ 的 `weight` 是 `[N, K/2]` 的 int8 字节，每字节低、高半字节依次对应相邻输入列；`weight_scale` 与 `weight_zeros` 为 `[K/group_size, N]`。通用 GPTQ/AWQ 文件不一定使用此排列，不能只改文件名或元数据就导入。

SVDQuant 使用 `format: "svdquant_w4a4"`，还必须包含 `weight_proj_down`、`weight_proj_up`、`weight_smooth_factor`，并按校准配置设置 `act_unsigned` 等参数。支持 Kitchen 的自然矩阵与 N-interleaved 分块存储；不接受运行时专用的 `mma_packed` 或转置存储。已有 `.comfy_quant` 的规范文件可直接读取，无需再次提供 JSON。请求相同的已校准目标格式会保留其数据；对浮点输入请求这两种格式则报错。

## 文件、磁盘空间与结果

输入与输出使用不同路径。除已有文件外，建议预留约输出大小两倍的临时磁盘空间。单层权重及其附属数据需要能放入所选设备。

成功后命令会打印 JSON，报告转换数量、保留的量化层和跳过原因，使用前请检查。反量化不能恢复量化时丢失的信息。移动或分享量化文件时应保留完整元数据。

## 运行时兼容性

UniTuner 支持使用上表中无需校准的格式进行适配器训练、预览和生成，全参数训练需要浮点权重。详见[量化基模训练](training/quantized-base.md)。

用于其他应用时，需确认目标加载器同时支持该模型和量化格式。低位权重存储不代表一定加速或支持 CPU，不同运行时的输出也可能不同。AWQ／SVDQuant 还需要支持对应校准格式的加载器。


---

[文档首页](README.md) · [参数参考](training/parameters.md) · [故障排查](troubleshooting.md)
