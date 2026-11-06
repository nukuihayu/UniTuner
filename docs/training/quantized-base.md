# 量化权重基模训练

简体中文 · [English](../en/training/quantized-base.md) · [训练指南](README.md)

UniTuner 支持在冻结的量化基模上训练 LoRA、LoHa 和 LoKr。量化可以降低基模权重的内存占用，但仍需要临时浮点计算空间，也可能增加耗时。选择较低精度前请比较预览效果。全参数训练需要浮点权重。

<details>
<summary>本页导航</summary>

- [训练前逐层量化](#训练前逐层量化)
- [可选精度](#可选精度)
- [读取已经量化的基模](#读取已经量化的基模)
- [显存与效果](#显存与效果)
- [保存与恢复](#保存与恢复)

</details>

## 训练前逐层量化

安装依赖后，在已有训练配置中设置量化精度，`model.weights` 继续指向浮点原生文件：

```bash
uv sync --locked
```

```toml
[train]
parameter_mode = "adapter"

[train.adapter]
algorithm = "lora" # 也支持 loha、lokr
rank = 32
alpha = 32.0

[execution]
quantization = "float8_e4m3fn"
precision = "bf16"
recompute = "block"
```

这是配置修改片段；保留原配置中的模型、数据、优化器、步数和输出路径。源文件可为 BF16、FP16 或 FP32。源文件不会被修改。

```bash
uv run unituner cache train.toml --device cuda
uv run unituner train train.toml --device cuda
```

准备量化基模时不要求完整浮点模型同时放入 GPU，但主机内存仍需容纳浮点源模型。训练时还应为激活和优化器状态预留空间。

## 可选精度

无需校准的布局均可用于权重存储：

| 配置值 | 权重存储 |
| --- | --- |
| `float8_e4m3fn`、`float8_e5m2` | 张量级缩放 FP8 |
| `mxfp8` | 分块缩放 FP8 |
| `nvfp4` | 分块缩放 FP4 |
| `int8_tensorwise`、`int8_rowwise` | 张量级或按输出行缩放 INT8 |
| `int8_convrot` | FP32 旋转后按行量化的 INT8 |
| `convrot_w4a4`、`convrot_w4a8` | ConvRot 4 位权重布局 |
| `asym_w4a8_int8`、`w6a8_int8` | 分组 4／6 位权重布局 |

名称中的 A4／A8 是上游布局名称，在 UniTuner 中仍使用浮点激活。`fp8` 是 `float8_e4m3fn` 的别名，`convrot-int8` 是 `int8_convrot` 的别名；它们只选择权重格式。AWQ／SVDQuant 需要离线校准，不参与训练前自动量化，原生训练加载器可读取已经离线校准的规范 AWQ／SVDQuant 基模。

ConvRot 默认旋转组为 256，可通过 `quantization_rotation_size` 调整；INT8 ConvRot 的组大小必须为至少 4 的 4 的幂，且整除投影输入宽度。分组格式通过 `quantization_group_size` 设置组大小，默认 16。尺寸不兼容会明确报错。需要选择性保留不兼容层时，先使用独立转换命令及 `--on-incompatible keep`。

## 读取已经量化的基模

也可以提前转换为可复用文件：

```bash
uv run unituner weights convert models/qwen21.safetensors models/qwen21-fp8.safetensors \
  --family qwen-image-2-1 --precision float8_e4m3fn --device cuda
```

将 `model.weights` 改为该文件。原生加载器根据 `*.comfy_quant` 自动识别量化层，`quantization = "none"` 表示保留文件中的存储布局。显式选择精度时，已有量化层必须匹配，禁止隐式重复量化有损权重。

转换保留文件键与拓扑，不能将任意外部 checkpoint 自动变为原生模型。需要格式导入时先导入浮点文件，再量化。支持的模型与格式详见[权重转换](../weight-conversion.md)。

## 显存与效果

敏感层会自动保留较高精度，常规训练无需逐层配置。量化主要减少基模权重存储，激活和临时计算仍占用显存。可结合 BF16 计算及支持的卸载、重算设置，参见[精度设置](precision.md)。

适配器按浮点权重训练和导出。请用验证预览检查量化是否影响细节、颜色或提示词响应。

## 保存与恢复

输出仍为浮点适配器；基模编码不会混入适配器文件。续训身份包含源文件与执行配置，使用相同源文件、精度、量化参数及设备环境重新建立基模。改变这些条件应视为不同训练设置。

预览直接使用训练中的模型。独立生成时若训练前自动量化，应先用相同设置转换出基模文件，再将生成配置指向它；直接使用原浮点基模可能改变输出。不要把适配器增量直接加到整数或打包编码上。

文本编码器使用独立的 `model.text.quantization`，可在 conditioning 缓存或推理前量化，也能读取打包文件；参阅[文本编码精度与量化](text-encoding.md)。

---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
