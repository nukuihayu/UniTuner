# 量化权重基模训练

简体中文 · [English](../en/training/quantized-base.md) · [训练指南](README.md)

UniTuner 支持冻结量化基模并训练 LoRA、LoHa 或 LoKr。所有精度共用 `QuantizedLinear`：只量化权重，前向临时恢复当前层的浮点权重，反向重新恢复同一权重并计算输入梯度。输入保持浮点，不进行动态激活量化，也不调用整数矩阵乘法。

这会降低基模的常驻存储和块传输量，但当前层仍需临时浮点权重，前向与反向还会增加反量化开销。它不保证加速，也不保证与原始浮点基模训练结果相同；梯度对应的是反量化后实际执行的浮点线性运算。

<details>
<summary>本页导航</summary>

- [训练前逐层量化](#训练前逐层量化)
- [可选精度](#可选精度)
- [读取已经量化的基模](#读取已经量化的基模)
- [计算与保护](#计算与保护)
- [保存与恢复](#保存与恢复)

</details>

## 训练前逐层量化

安装依赖后，在已有训练配置中设置量化精度，`model.weights` 继续指向浮点原生文件：

```bash
uv sync --locked --extra kitchen
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

这是配置修改片段；保留原配置中的模型、数据、优化器、步数和输出路径。源文件可为 BF16、FP16 或 FP32。原生训练加载器建立 FP32 基模，然后在适配器、优化器和执行策略安装前量化。

```bash
uv run unituner cache train.toml --device cuda
uv run unituner train train.toml --device cuda
```

使用 CUDA 训练时，浮点基模先保留在 CPU，每次只把一个投影送入 GPU 量化，再将量化结果放回 CPU。全部准备好后，resident 模式将量化模型移入 GPU；frozen-blocks／VBAR 按各自的驻留策略传输。完整浮点基模不会先占用 GPU，但仍需要相应的主机内存。CPU 训练则在 CPU 量化。

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

转换保留文件键与拓扑，不能将任意外部 checkpoint 自动变为原生模型。需要格式导入时先导入浮点文件，再量化。十个基模家族共享同一加载路径和保护策略，详见[权重转换](../weight-conversion.md)。

## 计算与保护

按家族保留输入输出、时间与条件投影、归一化和调制等敏感层。Z-Image 的 noise/context refiner 保持浮点。独立转换与训练前量化使用同一策略，包含主块内部的控制、运动、人脸、音频等辅助路径保护。训练前加载建立 FP32 基模，因此跳过的敏感层保留 FP32 参数；独立文件量化则保留源 dtype。跳过量化本身不禁用 AMP，详见[模型精度保护](precision.md)。

打包权重、缩放与校正张量全部冻结；只有浮点适配器进入优化器。模型 dtype 迁移保持各布局原有的存储与缩放 dtype，不能将 FP32 scale 或 FP8 block scale 一律转换成 BF16。适配器使用投影的逻辑形状，不能把打包后的存储宽度当成输入宽度。

INT8 ConvRot 的权重编码遵循 Musubi 的 FP32 旋转约定。计算时按布局恢复到原始坐标系，随后执行普通浮点线性层；不使用 Musubi 的动态激活量化或近似反向。反量化固定使用上游 eager 实现，并在张量所在设备运行，保证前向、反向和重算不会因线程中的后端选择不同而恢复出不同的低精度权重。

基模反向不保存用于权重梯度的输入，也不长期保存完整浮点基模。冻结块重算会保存和重新使用对应的量化存储，避免反向错误读取已卸载到 CPU 的模块属性。BF16 AMP、编译外围计算和冻结块卸载仍使用既有执行设置。全参数训练要求浮点权重。

## 保存与恢复

输出仍为浮点适配器；基模编码不会混入适配器文件。续训身份包含源文件与执行配置，使用相同源文件、精度、量化参数及设备环境重新建立基模。改变这些条件应视为不同训练设置。

预览直接使用训练中的模型。独立生成时若训练前自动量化，应先用相同设置转换出基模文件，再将生成配置指向它；直接使用原浮点基模可能改变输出。不要把适配器增量直接加到整数或打包编码上。

文本编码器使用独立的 `model.text.quantization`，可在 conditioning 缓存或推理前量化，也能读取打包文件；参阅[文本编码精度与量化](text-encoding.md)。

---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
