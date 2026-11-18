# 训练与导出精度

简体中文 · [English](../en/training/precision.md) · [训练指南](README.md)

训练计算、文本编码和导出文件可以分别设置精度。先使用默认 `auto`；需要明确控制显存和兼容性时，再选择下表中的模式。

## 选择训练精度

| `execution.precision` | 适用情况 | 注意事项 |
| --- | --- | --- |
| `auto` | 默认，按底模权重选择 FP32、FP16 或 BF16 | 量化底模默认使用 BF16；CPU 训练请显式选择 FP32 |
| `fp32` | CPU 训练或需要 FP32 的配方 | 显存占用通常较高 |
| `bf16` | 支持 BF16 的 NVIDIA GPU | 通常适合作为 CUDA 训练的起点 |
| `fp16` | 需要 FP16 的 CUDA 配方 | 对数值范围更敏感，出现 NaN 时优先尝试 BF16 |

```toml
[execution]
precision = "bf16"
```

也可以在单次运行时覆盖：

```bash
unituner train train.toml --device cuda --precision bf16
```

全参数训练和适配器训练均可选择浮点计算精度。可训练参数和优化器使用 FP32；模型需要保持高精度的部分会自动处理，无需逐层配置。

## FP16 出现 NaN 时

先降低学习率并检查数据和损失权重。如果仍出现溢出，可降低 FP16 的损失缩放系数：

```toml
[execution]
precision = "fp16"
fp16_loss_scale = 64.0
```

默认缩放系数为 128。设备支持时，改用 BF16 通常更方便。BF16 不需要设置这一系数；出现非有限梯度时训练会停止，保留日志用于排查。

## 文本与修复特征精度

文本编码使用独立配置，不跟随 `execution.precision`：

```toml
[model.text]
precision = "bf16"
```

修改文本精度或量化设置后重新生成缓存。具体模型限制见[文本编码](text-encoding.md)。

修复训练的 DINO 特征网络默认使用 BF16；需要 FP32 时设置：

```toml
[method.restoration]
feature_precision = "fp32"
```

FP32 特征网络会增加显存占用。训练预览使用训练精度；比较效果时固定提示词、种子、尺寸和采样参数。

## 导出精度

只想减小推理文件时，设置导出精度即可：

```toml
[output]
save_precision = "bf16"
```

可选 `fp32`、`bf16`、`fp16`。这不会更改训练计算精度或续训状态的精度。推理工具需要支持导出文件使用的精度。

## 量化与续训

量化主要降低冻结底模的显存和存储需求，适用于 LoRA、LoHa、LoKr；全参数训练使用浮点权重。通过 `execution.quantization` 选择权重格式，使用 `execution.precision` 独立设置计算精度，见[量化基模训练](quantized-base.md)。

切换精度或量化格式后，原有续训状态可能不再兼容。比较不同设置时使用新的输出目录，并固定其他训练条件。

---

[文档首页](../README.md) · [参数参考](parameters.md) · [故障排查](../troubleshooting.md)
