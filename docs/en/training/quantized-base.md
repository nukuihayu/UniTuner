# Training with quantized base weights

[简体中文](../../training/quantized-base.md) · English · [Training guide](README.md)

UniTuner supports LoRA, LoHa, and LoKr training on a frozen quantized base. Quantization reduces base-weight memory, but needs temporary floating-point working memory and may increase runtime. Compare previews before choosing a lower precision. Full-parameter training requires floating weights.

<details>
<summary>On this page</summary>

- [Quantize each layer before training](#quantize-each-layer-before-training)
- [Storage precisions](#storage-precisions)
- [Load a reusable quantized base](#load-a-reusable-quantized-base)
- [Memory and quality](#memory-and-quality)
- [Saving and resuming](#saving-and-resuming)

</details>

## Quantize each layer before training

Sync the default dependencies, then add a storage precision to an existing training configuration. Keep `model.weights` pointing to the floating native artifact:

```bash
uv sync --locked
```

```toml
[train]
parameter_mode = "adapter"

[train.adapter]
algorithm = "lora" # loha and lokr also supported
rank = 32
alpha = 32.0

[execution]
quantization = "float8_e4m3fn"
precision = "bf16"
recompute = "block"
```

This is an amendment, not a complete configuration. Keep the original model, data, optimizer, update count, and output settings. Source weights may be BF16, FP16, or FP32. The source files are not modified.

```bash
uv run unituner cache train.toml --device cuda
uv run unituner train train.toml --device cuda
```

Preparation does not require the complete floating base to fit in GPU memory, but host RAM must still accommodate the floating source model. Reserve extra memory for training activations and optimizer states.

## Storage precisions

All calibration-free layouts are supported:

| Configuration value | Weight storage |
| --- | --- |
| `float8_e4m3fn`, `float8_e5m2` | Tensor-scaled FP8 |
| `mxfp8` | Block-scaled FP8 |
| `nvfp4` | Block-scaled FP4 |
| `int8_tensorwise`, `int8_rowwise` | Tensor-scaled or output-row-scaled INT8 |
| `int8_convrot` | Row-scaled INT8 after FP32 rotation |
| `convrot_w4a4`, `convrot_w4a8` | ConvRot 4-bit weight layouts |
| `asym_w4a8_int8`, `w6a8_int8` | Grouped 4/6-bit weight layouts |

A4/A8 are upstream layout names; UniTuner still uses floating activations. `fp8` aliases `float8_e4m3fn`, and `convrot-int8` aliases `int8_convrot`. These names select weight storage only. AWQ/SVDQuant require offline calibration and are excluded from automatic preparation; native training can load existing calibrated canonical artifacts.

The default rotation group is 256; change it with `quantization_rotation_size`. INT8 ConvRot requires a power of four of at least 4 dividing the projection's input width. Grouped formats use `quantization_group_size`, default 16. Incompatible dimensions fail explicitly. To preserve incompatible projections selectively, convert separately with `--on-incompatible keep`.

## Load a reusable quantized base

You can prepare a reusable artifact separately:

```bash
uv run unituner weights convert models/qwen21.safetensors models/qwen21-fp8.safetensors \
  --family qwen-image-2-1 --precision float8_e4m3fn --device cuda
```

Point `model.weights` to that artifact. Native loaders recognize `*.comfy_quant` automatically. `quantization = "none"` retains the artifact's existing storage. An explicitly requested precision must match already quantized layers; implicit lossy re-quantization is rejected.

Conversion preserves keys and topology; it does not turn arbitrary external checkpoints into native models. Import floating weights first when necessary, then quantize. For supported models and formats, see [weight conversion](../weight-conversion.md).

## Memory and quality

Sensitive layers automatically retain higher precision. No manual layer selection is needed for ordinary training. Quantization affects base-weight storage; activations and temporary working space still consume memory. You can combine it with BF16 computation and supported offload/recomputation settings; see [precision settings](precision.md).

Adapters train and export as floating-point weights. Use held-out previews to check whether quantization changes detail, color, or prompt adherence.

## Saving and resuming

Outputs remain floating adapters; packed base codes are excluded. Resume identity includes source files and execution configuration. Recreate the base using the same sources, precision, quantization options, and device environment. Changes constitute a different training setup.

Previews use the live training model. For standalone generation after automatic preparation, convert the base with matching settings and point generation to that artifact. Using the original floating base may change outputs. Never add adapter deltas directly to integer or packed codes.

Text encoders use independent `model.text.quantization` settings, supporting quantization before conditioning caches/inference and packed artifact loading. See [text encoding](text-encoding.md).

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
