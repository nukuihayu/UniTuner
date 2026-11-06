# Training and export precision

[简体中文](../../training/precision.md) · English · [Training guide](README.md)

Set training computation, text encoding and export precision independently. Start with the default `auto`, then select an explicit mode when you need to control memory or compatibility.

## Choose training precision

| `execution.precision` | Use | Notes |
| --- | --- | --- |
| `auto` | Default; selects FP32, FP16 or BF16 from the base weights | Quantized bases default to BF16; choose FP32 explicitly for CPU |
| `fp32` | CPU training or an FP32 recipe | Usually needs more GPU memory |
| `bf16` | NVIDIA GPUs with BF16 support | A useful starting point for CUDA training |
| `fp16` | CUDA recipes requiring FP16 | More sensitive to numeric range; try BF16 if NaNs occur |

```toml
[execution]
precision = "bf16"
```

Override it for one run:

```bash
unituner train train.toml --device cuda --precision bf16
```

Full and adapter training support floating compute precisions. Trainable parameters and optimizer state use FP32. Model operations that need higher precision are handled automatically; no per-layer configuration is needed.

## If FP16 produces NaNs

First reduce the learning rate and check data and loss weights. If overflow persists, lower the FP16 loss scale:

```toml
[execution]
precision = "fp16"
fp16_loss_scale = 64.0
```

The default scale is 128. Switching to BF16 is usually easier when supported by the device. BF16 needs no loss-scale setting. Training stops on nonfinite gradients; retain the logs for diagnosis.

## Text and restoration features

Text encoding has its own precision settings:

```toml
[model.text]
precision = "bf16"
```

Rebuild caches after changing text precision or quantization. See [text encoding](text-encoding.md) for model-specific restrictions.

Restoration DINO/VGG feature networks default to BF16. Select FP32 with:

```toml
[method.restoration]
feature_precision = "fp32"
```

FP32 feature networks need more GPU memory. Training previews use training precision; keep prompts, seeds, dimensions and sampling settings fixed when comparing results.

## Export precision

To reduce inference file size, change export precision:

```toml
[output]
save_precision = "bf16"
```

Choose `fp32`, `bf16` or `fp16`. Export precision does not change training computation or resume-state precision. The inference tool must support the exported precision.

## Quantization and resume

Quantization reduces frozen-base storage and memory for LoRA, LoHa and LoKr. Full training uses floating weights. Set weight format with `execution.quantization` and compute precision independently with `execution.precision`; see [quantized-base training](quantized-base.md).

Changing precision or quantization can make an existing resume state incompatible. Use a new output directory when comparing settings and keep other training conditions fixed.

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
