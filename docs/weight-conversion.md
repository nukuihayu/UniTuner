# 权重精度转换

简体中文 · [English](en/weight-conversion.md) · [文档首页](README.md)

`unituner weights convert` 将 safetensors 权重转换为普通浮点或 comfy-kitchen 量化布局。它直接调用 Kitchen 的量化与序列化接口，保留每层需要的缩放、旋转、码本和校正数据，不自行实现另一套打包规则。

转换默认导出 ComfyUI 主模型布局：自动识别 Diffusers／原生权重，完成键名、QKV 和调制参数转换后再处理精度。按块读取和转换，不分配完整模型。使用 `--output-format source` 可保留输入布局。

<details>
<summary>本页导航</summary>

- [开始转换](#开始转换)
- [支持的精度](#支持的精度)
- [按模型保护敏感层](#按模型保护敏感层)
- [选择需要转换的层](#选择需要转换的层)
- [导入 AWQ 与 SVDQuant](#导入-awq-与-svdquant)
- [文件与数值约定](#文件与数值约定)
- [运行时兼容性](#运行时兼容性)

</details>

## 开始转换

在项目根目录安装可选依赖：

```bash
uv sync --locked --extra kitchen
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

Qwen Image 2.1 可直接读取 ComfyUI 的独立 `gate_layer`／`proj` 或融合 `gate_up` 权重，以及常用组件前缀和索引分片。无拓扑元数据时，从权重头识别层数、通道、注意力头数与 MLP 比例；RoPE 轴分配和 epsilon 无法从权重推断，自定义值需在 `config.json` 或显式拓扑中提供。融合权重按 `[gate; up]` 拆分。INT8 ConvRot 等量化文件仅将融合 MLP 输入投影还原为浮点后拆分，其余已量化投影保留原始存储；因此载入后的内存占用会高于全量化文件大小。ComfyUI 保存的 `comfy_attention.config` 作为推理后端偏好校验后过滤，训练使用自身的注意力设置。缺失权重、冲突布局或不一致的结构会报错，不会随机补齐参数。

`weights import --kind transformer` 也默认导出 ComfyUI 布局；自定义结构可提供 `--topology`，需要原生布局时指定 `--output-format native`。

## 支持的精度

| `--precision` | Kitchen 布局 | 转换方式 |
| --- | --- | --- |
| `fp32`、`fp16`、`bf16` | 普通浮点 | 浮点转换；量化输入先反量化 |
| `float8_e4m3fn`、`float8_e5m2` | `TensorCoreFP8Layout` | 张量级缩放 FP8 |
| `mxfp8` | `TensorCoreMXFP8Layout` | 每 32 元素 E8M0 缩放 |
| `nvfp4` | `TensorCoreNVFP4Layout` | 每 16 元素 FP8 缩放，加 FP32 全局缩放 |
| `int8_tensorwise` | `TensorWiseINT8Layout` | 张量级 INT8 |
| `int8_rowwise` | `TensorWiseINT8Layout` | 按输出行缩放 INT8 |
| `int8_convrot` | `TensorWiseINT8Layout` | 按行缩放，附带 ConvRot |
| `convrot_w4a4`、`convrot_w4a8` | `TensorCoreConvRotW4A4Layout` | 4 位权重，4 位或 8 位激活布局 |
| `asym_w4a8_int8` | `AsymW4A8Int8Layout` | 分组 4 位码本权重 |
| `w6a8_int8` | `AsymW4A8Int8Layout` | 分组 6 位均匀权重 |
| `awq_w4a16` | `TensorCoreAWQW4A16Layout` | 导入已离线校准的权重 |
| `svdquant_w4a4` | `TensorCoreSVDQuantW4A4Layout` | 导入已离线校准的权重及低秩分支 |
| `keep` | 保持原布局 | 校验并重新封装，保留量化数据 |

W/A 描述部署时的权重与激活精度。文件转换保存权重和布局参数；激活量化由消费该文件的推理运行时执行。

## 按模型保护敏感层

默认 `--family auto` 优先读取原生拓扑元数据。没有元数据时，根据家族特有的键识别；键结构不明确时显式指定 `--family`。家族名称与训练一致：`z-image`、`flux`、`flux2`、`flux3`、`moss-image`、`qwen-image`、`qwen-image-2-1`、`krea2`、`wan`、`minimax-h3`。

```bash
uv run unituner weights convert models/qwen21.safetensors models/qwen21-int8.safetensors \
  --family qwen-image-2-1 --precision int8_convrot --device cuda --on-incompatible keep
```

| 模型 | 默认量化范围 | 主要浮点保护 |
| --- | --- | --- |
| Z-Image | 主 `layers` | noise/context refiner、输入输出、时间和文本嵌入、norm、modulation |
| FLUX／FLUX.2 | double/single blocks | 输入输出、时间／guidance、norm、modulation |
| FLUX3 | 内容／文本早期块、single blocks | 模式输入输出、时间／目标时间、norm、modulation |
| MOSS Image | 主 `layers` | caption refiner、输入输出、时间、norm、AdaLN |
| Qwen Image／2.1 | `transformer_blocks` | 输入输出、时间与上下文投影、norm、modulation |
| Krea2 | 主 `blocks` | text fusion、输入输出、时间、norm、mod |
| Wan | 主 `blocks` | patch/text/time、输出 head、norm、modulation |
| MiniMaxH3 | 注意力 qkv/out、MLP fc1/fc2、块内 adaln linear | token refiner、最终输出、输入投影与 norm |

所有家族共同排除 norm、bias、词表和调制，以及控制、运动、人脸、音频和多模态辅助分支，即使它们嵌套在主块内部。Krea2 的 `first`、`last`、`tmlp`、`txtfusion`、`txtmlp` 和 `tproj` 保持浮点。量化时受保护张量保留源 dtype，包括已有 FP32 权重；不会把 BF16／FP16 源数据假装恢复为原始 FP32 精度。显式 `--precision bf16` 等普通浮点转换仍按请求转换选中的浮点张量。

这些是保守部署策略，不是所有层量化后质量等价的承诺。`--include` 只能进一步缩小范围，不能覆盖保护；报告中的 `skipped` 同时列出保护原因和尺寸不兼容原因。`--family generic` 适合不在列表中的模型，按二维权重规则选择，同时保护名称中可识别的 norm、embedding、time 与 modulation；它不能代替未知模型的架构审查。`--output-format source` 保留原文件键名；已知外层扩散前缀在保护策略中按模型内部名称匹配。

### 文本编码器保护

显式选择 `--family qwen2-5-vl|qwen3|qwen3-vl|qwen3-5|mistral|t5|umt5|clip`。这些策略覆盖当前项目使用的 Qwen2.5-VL、Qwen3、Qwen3-VL、Qwen3.5、Mistral、T5／UMT5 和 CLIP。它们按实际数字层号保护每个编码器栈的首层和末层，同时保留 embedding、LM head、共享词表、relative attention bias、归一化、完整视觉／音频分支、multimodal projector 与 MTP。Qwen3.5 的 `in_proj_a`／`in_proj_b` 也保持浮点。

```bash
uv run unituner weights convert models/text_encoder/model.safetensors models/text_encoder-fp8.safetensors \
  --family qwen3-vl --precision float8_e4m3fn --device cuda
```

保护只影响量化选择，不会删除 decoder、LM head 或其他源文件键。首末层按完整分片清单计算，不能逐分片猜测。外部语言模型结构通常相似，`auto` 无法可靠区分时必须指定家族。文本编码器文件转换不改变各家族现有 tokenizer、配置和命名空间要求。原生文本加载器可直接读取全部无需校准布局，并用于 conditioning 缓存、预览和推理；也可在编码前逐层量化浮点源。配置与各模型的文本资产路径见[文本编码精度与量化](training/text-encoding.md)。

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
| `--family` | `auto` | 家族保护策略；含 `generic`、十个基模家族和八类文本编码器 |
| `--device` | `cpu` | 转换使用的 `cpu` 或 `cuda` 设备 |
| `--include` / `--exclude` | 无 | 一个或多个参数键 glob |
| `--rotation-size` | `256` | ConvRot 的旋转组大小，必须为至少 4 的 4 的幂 |
| `--group-size` | `16` | 分组 4/6 位布局的量化组大小，至少为 4，且整除 16 或为 16 的倍数 |
| `--on-incompatible` | `error` | 不满足布局形状要求时终止；`keep` 保留该层浮点权重，并在结果中列出原因 |
| `--backend` | 自动 | 可选 `eager`、`cuda`、`triton`；后两者需满足 Kitchen 的硬件与内核条件 |
| `--layout-spec` | 无 | 为规范打包的离线权重补充布局描述 |

FP32 输入的 MXFP8／NVFP4 使用同设备上的 Kitchen eager 量化，避免原生内核仅支持 FP16／BF16 而被迫提前降精度。INT8 ConvRot 在 FP32 中旋转并确定编码，即使源权重为 BF16／FP16。CPU 可使用 `--backend eager`，但量化方法、矩阵尺寸与计算 dtype 仍需满足上游约束。`--on-incompatible keep` 不会绕过缺失校准、损坏元数据或运行时内核错误。

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

## 文件与数值约定

每层通过 uint8 UTF-8 JSON 张量 `*.comfy_quant` 保存 `format`、逻辑 `orig_shape`、`orig_dtype` 及布局参数。布局自身负责生成附属键：FP8/INT8 使用 `weight_scale`，NVFP4 另含 `weight_scale_2`；分组布局保留 `weight_s_rel`、`weight_s_channel` 及可选校正、码本；AWQ/SVDQuant 保留上述校准张量。INT8 的按行、旋转选项和 W4A8 的 `linear_dtype` 会写入元数据。

MXFP8 的 E8M0 缩放以原始 uint8 字节保存，以兼容 safetensors；重载时按原 dtype 解释。分块缩放顺序、填充和位打包均遵循 Kitchen。逻辑形状与填充后的存储形状分别保留，因此非整块尺寸反量化后仍恢复原始尺寸。缺失逻辑形状的旧打包文件需用 `--layout-spec` 补齐。

已有量化层被排除或使用 `keep` 时，会保留其附属数据、`input_scale` 与 `pre_quant_scale`。重新量化或转为浮点时，输入平滑 `pre_quant_scale` 会乘入权重的输入列，旧激活缩放不再沿用。反量化不能恢复量化前丢失的信息；SVDQuant 的浮点导出使用上游有效权重重建，也不能完整复现任意输入上的非线性激活量化。

转换逐张量读取并暂存输出数据，不把整个模型放入内存；单层及其附属张量仍需能放入所选设备。完成前输出以临时文件构建，失败不会替换已有目标。临时磁盘空间约需输出大小的两倍，另计已有文件。成功后打印 JSON，报告转换数、保留的量化层数及跳过原因；文件头记录转换选项和 Kitchen 版本。

## 运行时兼容性

UniTuner 十个基模家族的原生加载器支持全部无需校准布局的适配器训练、预览和生成，也支持从浮点基模在 GPU 逐层量化后训练。所有格式只量化权重，计算时临时反量化并执行浮点乘法，不启用动态激活量化；见[量化基模训练](training/quantized-base.md)。全参数训练要求浮点权重。所有输入都需符合模型加载器的键与结构，必要时先用 `weights import` 导入浮点原生格式，再量化。

量化输出用于支持对应 Kitchen 布局的消费者。ComfyUI 可按布局执行激活量化，UniTuner 的权重量化计算则保留浮点激活；文件互读正确不代表两者输出逐位一致。核对时的 ComfyUI 加载器支持表中的直接量化格式，但尚无 AWQ/SVDQuant 加载分支；这两个布局能在 Kitchen 中重载，不代表当前 ComfyUI 可直接加载。E5M2 文件会记录 `full_precision_matrix_mult: true`，让 ComfyUI 反量化后做浮点乘法，避免 PyTorch 不支持 E5M2×E5M2 的内核报错。实际推理还取决于模型加载器、GPU、dtype 和内核；例如 MXFP8／NVFP4 的实际计算需要相应 GPU 路径，不能据此推断 CPU 支持。

---

[文档首页](README.md) · [参数参考](training/parameters.md) · [故障排查](troubleshooting.md)
