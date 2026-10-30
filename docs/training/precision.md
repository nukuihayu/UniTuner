# 模型精度保护

简体中文 · [English](../en/training/precision.md)

[训练指南](README.md) · [执行策略](execution.md)

## 存储、计算和导出精度

可训练参数与 optimizer masters 保留 FP32。冻结基模按自动或显式精度加载，也可使用 Kitchen 量化布局。三个选项分别控制不同阶段：

| 选项 | 作用 |
| --- | --- |
| `execution.quantization` | 训练前逐层量化冻结基模；`none` 保留输入文件的布局 |
| `execution.precision` | 自动／显式浮点存储与计算精度，或 Kitchen 量化格式选择 |
| `output.save_precision` | 发布时转换普通浮点张量；续训状态保留训练 dtype |

所有量化格式共用临时反量化后的浮点线性运算，输入保持浮点。独立转换和训练前量化使用同一套敏感层预设；`include` 不能覆盖保护。独立量化保留受保护张量的源 dtype，训练前量化则在原生加载器建立 FP32 基模后执行，因此受保护路径保留 FP32 参数。跳过量化不等于禁用 AMP，也不能恢复源文件已经丢失的精度。


默认 `execution.precision = "auto"`，按基模各分片中占元素数最多的浮点矩阵精度选择 FP32、FP16 或 BF16；量化基模默认 BF16 计算。只读取文件头，不把整份权重载入显存后再判断。显式 TOML 设置优先于自动判断，CLI 参数覆盖 TOML，多个 CLI 精度参数按最后一次赋值生效。

```bash
unitunner train train.toml --device cuda --precision bf16
unitunner train train.toml --device cuda --precision fp16
unitunner train train.toml --device cuda --precision int8_rowwise
```

`--precision` 是 `--execution.precision` 的别名，也可在 TOML 的 `[execution]` 写 `precision`。可选 `auto`（`keep` 同义）、`fp32`、`fp16`、`bf16`，以及 Kitchen 的 `float8_e4m3fn`、`float8_e5m2`、`mxfp8`、`nvfp4`、`int8_tensorwise`、`int8_rowwise`、`int8_convrot`、`convrot_w4a4`、`convrot_w4a8`、`asym_w4a8_int8`、`w6a8_int8`。Kitchen 名称映射为权重量化选择，浮点计算仍由源文件自动判断；若需独立指定计算精度，分别设置 `execution.quantization` 和 `execution.precision`。训练精度和量化入口不接受需要离线校准的 AWQ／SVDQuant；独立权重转换工具仍支持导入已校准文件。量化训练只支持冻结基模的 adapter 模式。

Adapter 模式的普通冻结底模直接以所选 BF16／FP16／FP32 存储；全参数训练保留 FP32 主参数，并按选择使用 AMP。Adapter 因子也保留 FP32。模型必要的 FP32 时间／投影区域继续保留，量化码和 scale 不做普通 dtype 转换。修复的 VGG、DINO 默认以 BF16 加载并使用 BF16 AMP，独立于 denoiser 精度；可用 `method.restoration.feature_precision = "fp32"` 覆盖。VAE 的修复卷积计算跟随 denoiser 精度，参数仍保留 FP32。

训练预览的模型计算与采样状态跟随 `execution.precision`，支持 FP32、BF16、FP16，不要求底模权重全部为 FP32，也不会临时转换训练权重。图像、视频及音视频共用这一策略；原生算子必要的高精度计算和 VAE 解码在各自边界处理。直接调用底层采样器时，输出状态保留传入噪声的浮点 dtype，可独立使用 FP32、BF16 或 FP16。预览结束或失败后会恢复训练模式、随机数状态与精度作用域。

FP16 使用固定 loss scale，默认 `execution.fp16_loss_scale = 128.0`，反向后先取消缩放，再按样本数归一化并裁剪。非有限梯度会在任何 optimizer 更新前报错，可显式降低 loss scale；不自动跳过 batch。BF16 不需要 loss scaling。浮点存储策略与缩放设置参与续训身份，恢复训练时必须通过相应身份检查。

## 原生精度保护

归一化、位置几何、时间条件和部分投影保留模型要求的精度；卸载与重算遵循同样规则。

| 家族 | 原生处理 |
| --- | --- |
| FLUX | 冻结参数可用低精度存储；Q/K RMSNorm 和 RoPE 用 FP32 运算，再恢复激活 dtype |
| FLUX.2 | 原生位置矩阵 FP32；RMSNorm 用 FP32；调制保持原生计算顺序 |
| FLUX3 | Q/K 归一化与旋转显式禁用 autocast，保持 FP32 累加和原生位置精度 |
| MOSS Image | RMSNorm 用 FP32，时间与位置构造保留原生 dtype |
| Qwen Image | 位置相位／旋转用原生浮点计算，RMSNorm 平方归约 FP32；保持普通 RMSNorm 的乘法顺序 |
| Qwen Image2.1 | 归一化含零中心 scale，即 `1 + weight`，用 FP32；保留与上一代不同的位置和文本处理 |
| Z-Image | 两个敏感嵌入器保持浮点存储；RMSNorm 用 FP32，位置频率按原生 FP64 构造后转 FP32；skip 不等于禁用嵌入器 AMP |
| Krea2 | 对应原生 `RMSNorm.scale`／QKNorm 子模块；保持 FP32 的 `scale + 1` 和 RMSNorm，输出恢复激活 dtype |
| Wan | 对应 `time_embedding`, `time_projection`, `blocks.*.modulation`, `head.modulation` 与 norm；时间投影显式禁用 AMP，norm／调制保留 FP32。skip 中的条件嵌入对应原生 text/image embedding，不代表它们全部禁用 AMP |
| MiniMaxH3 | 显式保护输入 video/audio patch projection、时间 MLP、曲线表／SVD basis、输出 video/audio head 和 RoPE；AdaLN 投影属于块计算，不因名称含 `proj` 就全部升 FP32 |

## 量化层预设

所有家族共同保护 norm、bias、embedding／共享词表、调制，以及控制、运动、人脸、音频和多模态辅助分支；这些分支嵌套在主 Transformer 块中时也会被排除。家族规则进一步保护条件、时间和输出路径：Z-Image 保留两个 refiner，MOSS Image 保留 caption refiner，Wan 保留 face adapter／audio injector。Krea2 保留 `first`、`last`、`tmlp`、`txtfusion`、`txtmlp` 和 `tproj`。MiniMaxH3 的块内 `adaln_proj.linear` 可以量化，不能因为名称含投影就把全部主块排除。

文本编码器预设保护各编码器栈的数字首末层、完整视觉／音频分支、词表、LM head、MTP 与敏感投影；不会删除源文件键。它们同时用于独立文件转换和编码前逐层量化。原生文本加载器支持全部无需校准布局；`model.text.precision` 控制浮点计算，`model.text.quantization` 控制权重存储，详见[文本编码精度与量化](text-encoding.md)。

完整范围与参数见[权重转换](../weight-conversion.md#按模型保护敏感层)；GPU 逐层准备、输入梯度和续训约束见[量化基模训练](quantized-base.md)。这些保护减少已知敏感路径的量化损失，不保证整个模型质量或训练轨迹与浮点基模相同。

---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
