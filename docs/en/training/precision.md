# Model precision safeguards

[简体中文](../../training/precision.md) · **English**

[Training](README.md) · [Execution](execution.md)

## Storage, computation, and publication precision

Trainable parameters and optimizer masters remain FP32. Frozen base storage follows automatic or explicit precision selection, with Kitchen quantized layouts also available. Three settings control separate stages:

| Setting | Purpose |
| --- | --- |
| `execution.quantization` | Quantize the frozen base layer by layer before training; `none` preserves the input artifact layout |
| `execution.precision` | Automatic or explicit floating storage/compute precision, or a Kitchen quantization selection |
| `output.save_precision` | Cast ordinary floating tensors at publication; resume state retains training dtypes |

Every quantized format uses temporary weight reconstruction and floating linear operations with floating inputs. Standalone conversion and pretraining quantization share layer safeguards; `include` cannot override them. Standalone quantization preserves protected tensors in their source dtype. Pretraining quantization follows native FP32 model loading, so protected parameters remain FP32. Skipping quantization does not disable AMP or recover precision already lost in source files.


The default `execution.precision = "auto"` selects FP32, FP16 or BF16 from the dominant floating matrix dtype by element count across all base-model shard headers. Quantized bases default to BF16 compute. Explicit TOML values override detection; CLI values override TOML, with the last CLI assignment winning.

```bash
unitunner train train.toml --device cuda --precision bf16
unitunner train train.toml --device cuda --precision fp16
unitunner train train.toml --device cuda --precision int8_rowwise
```

`--precision` aliases `--execution.precision`, also available as `precision` under `[execution]`. Choices include `auto` (`keep`), `fp32`, `fp16`, `bf16`, and Kitchen formats: `float8_e4m3fn`, `float8_e5m2`, `mxfp8`, `nvfp4`, `int8_tensorwise`, `int8_rowwise`, `int8_convrot`, `convrot_w4a4`, `convrot_w4a8`, `asym_w4a8_int8`, `w6a8_int8`. Kitchen choices select quantized storage with compute inferred from the source. Set `execution.quantization` and `execution.precision` separately for independent choices. Training precision and quantization inputs exclude AWQ/SVDQuant because they require offline calibration. Standalone weight conversion still supports importing calibrated files. Quantized training requires adapters over a frozen base.

Floating frozen adapter bases load directly into the selected BF16/FP16/FP32 storage. Full training and adapter factors retain FP32 master parameters while forward follows the selected AMP policy. Required native FP32 time/projection islands remain protected. Packed codes/scales retain their representation. Restoration VGG and DINO load and compute in BF16 by default independently of denoiser precision; override with `method.restoration.feature_precision = "fp32"`. Restoration VAE convolution compute follows denoiser precision while VAE parameter storage stays FP32.

FP16 uses fixed loss scaling, default `execution.fp16_loss_scale = 128.0`. Gradients are unscaled before sample normalization and clipping. Nonfinite gradients fail before any optimizer update; lower the configured scale if necessary. Batches are not silently skipped. BF16 needs no loss scaling. Storage and scaling policies participate in resume identity; resume requires a matching numerical identity.

## Native precision safeguards

Normalization, positional geometry, time conditioning, and selected projections retain their model-specific precision. Offload and recomputation preserve those policies.

| Family | Native behavior |
| --- | --- |
| FLUX | Floating frozen storage; Q/K RMSNorm and RoPE compute FP32, then restore activation dtype |
| FLUX.2 | FP32 position matrices/RMSNorm; native modulation order |
| FLUX3 | Follows official implementation; Q/K normalization and rotation explicitly disable autocast, preserving FP32 accumulation and native positions |
| MOSS Image | Follows official native model; FP32 RMSNorm, native time/position construction dtypes |
| Qwen Image | Native floating-point position phases/rotation; FP32 RMSNorm square reduction and native multiplication order |
| Qwen Image 2.1 | FP32 zero-centered normalization (`1 + weight`), with native position and text handling |
| Z-Image | Floating embedder storage; FP32 RMSNorm; native FP64 position frequencies converted to FP32; skip does not disable embedder AMP |
| Krea2 | Native `RMSNorm.scale` / QKNorm; FP32 `scale + 1` and RMSNorm, output restored to activation dtype |
| Wan | Native `time_embedding`, `time_projection`, `blocks.*.modulation`, `head.modulation`, and norms; time projection disables AMP; norms/modulation stay FP32; text/image embedding skip does not disable all their AMP |
| MiniMaxH3 | Explicit safeguards for video/audio patch projections, time MLP, curve tables/SVD basis, output heads, and RoPE; block AdaLN projections are not promoted solely for having `proj` in the name |

## Quantization presets

All families protect norms, biases, embeddings/shared vocabularies, modulation, and control, motion, face, audio, and multimodal auxiliary paths, including paths nested inside core Transformer blocks. Family rules additionally protect conditioning, time, and output branches: both Z-Image refiners, the MOSS Image caption refiner, and Wan face adapters/audio injectors remain floating. Krea2 retains `first`, `last`, `tmlp`, `txtfusion`, `txtmlp`, and `tproj`. MiniMaxH3 block-local `adaln_proj.linear` remains eligible; having a projection in its name does not exclude an entire core block.

Text presets protect each encoder stack's numeric first/last layers, complete vision/audio branches, vocabularies, LM heads, MTP, and sensitive projections without deleting source keys. These presets support both standalone conversion and quantization before encoding. Native text loaders accept every calibration-free layout. `model.text.precision` controls floating computation; `model.text.quantization` controls storage. See [text precision and quantization](text-encoding.md).

See [weight conversion](../weight-conversion.md#protect-sensitive-layers-by-model) for selection rules and [quantized-base training](quantized-base.md) for GPU preparation, input gradients, and resume constraints. These safeguards reduce quantization loss in known sensitive paths; they do not guarantee model quality or training trajectories identical to a floating base.

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
