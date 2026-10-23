# Training with quantized base weights

[简体中文](../../training/quantized-base.md) · English · [Training guide](README.md)

UniTunner trains LoRA, LoHa, or LoKr on frozen quantized bases. Every precision uses the same `QuantizedLinear`: reconstruct the current layer's floating weight for forward, reconstruct the same weight again for backward, and compute the input gradient. Activations remain floating point. There is no dynamic activation quantization or integer matrix multiplication.

This reduces resident base storage and block transfers. Each layer still needs a temporary floating weight, and both forward and backward incur reconstruction costs. It does not promise acceleration or equivalence to training the original floating base. Gradients correspond to the floating linear operation actually executed with reconstructed weights.

<details>
<summary>On this page</summary>

- [Quantize each layer before training](#quantize-each-layer-before-training)
- [Storage precisions](#storage-precisions)
- [Load a reusable quantized base](#load-a-reusable-quantized-base)
- [Computation and protections](#computation-and-protections)
- [Saving and resuming](#saving-and-resuming)

</details>

## Quantize each layer before training

Install the optional dependency, then add a storage precision to an existing training configuration. Keep `model.weights` pointing to the floating native artifact:

```bash
uv sync --locked --extra kitchen
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

This is an amendment, not a complete configuration. Keep the original model, data, optimizer, update count, and output settings. Source weights may be BF16, FP16, or FP32. Native training loaders materialize an FP32 base, then quantize it before installing adapters, optimizers, or execution owners.

```bash
uv run unitunner cache train.toml --device cuda
uv run unitunner train train.toml --device cuda
```

For CUDA training, floating weights initially stay on CPU. Transfer one projection to GPU, quantize it there, and return its packed representation to CPU. Once preparation finishes, resident execution moves the quantized model to GPU; frozen-blocks and VBAR apply their own placement policies. The complete floating base never needs to reside on GPU, but sufficient host memory is still required. CPU training quantizes on CPU.

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

A4/A8 are upstream layout names; UniTunner still uses floating activations. `fp8` aliases `float8_e4m3fn`, and `convrot-int8` aliases `int8_convrot`. These names select weight storage only. AWQ/SVDQuant require offline calibration and are excluded from automatic preparation; native training can load existing calibrated canonical artifacts.

The default rotation group is 256; change it with `quantization_rotation_size`. INT8 ConvRot requires a power of four of at least 4 dividing the projection's input width. Grouped formats use `quantization_group_size`, default 16. Incompatible dimensions fail explicitly. To preserve incompatible projections selectively, convert separately with `--on-incompatible keep`.

## Load a reusable quantized base

You can prepare a reusable artifact separately:

```bash
uv run unitunner weights convert models/qwen21.safetensors models/qwen21-fp8.safetensors \
  --family qwen-image-2-1 --precision float8_e4m3fn --device cuda
```

Point `model.weights` to that artifact. Native loaders recognize `*.comfy_quant` automatically. `quantization = "none"` retains the artifact's existing storage. An explicitly requested precision must match already quantized layers; implicit lossy re-quantization is rejected.

Conversion preserves keys and topology; it does not turn arbitrary external checkpoints into native models. Import floating weights first when necessary, then quantize. All ten base-model families share loading and protection logic; see [weight conversion](../weight-conversion.md).

## Computation and protections

Family policies preserve sensitive input/output, time/conditioning, normalization, and modulation paths. Z-Image noise/context refiners remain floating. Standalone conversion and training preparation share the same policy, including control, motion, face, and audio auxiliary paths nested inside core blocks. Training preparation starts from native FP32 loading, so protected parameters stay FP32; standalone quantization keeps source dtypes. Skipping quantization does not disable AMP; see [precision safeguards](precision.md).

Packed weights, scales, and correction tensors remain frozen; only floating adapter parameters enter the optimizer. Dtype migration preserves every layout's storage and scale dtype, including FP32 scales and FP8 block scales. Adapter geometry uses logical projection dimensions, not packed storage widths.

INT8 ConvRot encoding follows Musubi's FP32 rotation convention. Computation reconstructs weights in the original coordinate system before ordinary floating linear operations. It does not implement Musubi's dynamic activation quantization or approximate backward. Reconstruction consistently uses upstream eager operations on the tensor's device, preventing thread-local backend changes from reconstructing differently rounded weights during backward or replay.

Base backward does not save inputs for weight gradients or retain a complete floating base. Frozen-block replay reads the saved quantized storage rather than module attributes that may already have returned to CPU. BF16 AMP, compilation around quantized operators, and frozen-block offloading use existing execution settings. Full-parameter training requires floating weights.

## Saving and resuming

Outputs remain floating adapters; packed base codes are excluded. Resume identity includes source files and execution configuration. Recreate the base using the same sources, precision, quantization options, and device environment. Changes constitute a different training setup.

Previews use the live training model. For standalone generation after automatic preparation, convert the base with matching settings and point generation to that artifact. Using the original floating base may change outputs. Never add adapter deltas directly to integer or packed codes.

Text encoders use independent `model.text.quantization` settings, supporting quantization before conditioning caches/inference and packed artifact loading. See [text encoding](text-encoding.md).

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
