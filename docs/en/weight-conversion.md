# Weight precision conversion

[简体中文](../weight-conversion.md) · English · [Documentation](README.md)

`unitunner weights convert` converts safetensors weights to ordinary floating point or comfy-kitchen quantized layouts. It calls Kitchen's quantization and serialization APIs directly, retaining each layout's scales, rotations, codebooks, and corrections rather than maintaining a separate packing implementation.

Conversion exports ComfyUI denoiser layouts by default. It recognizes published Diffusers/native keys, converts QKV projections and modulation ordering, then applies precision conversion. Processing is bounded by transformer blocks without allocating full model storage. Use `--output-format source` to retain the input layout.

<details>
<summary>On this page</summary>

- [Convert a model](#convert-a-model)
- [Supported precisions](#supported-precisions)
- [Protect sensitive layers by model](#protect-sensitive-layers-by-model)
- [Select layers](#select-layers)
- [Import AWQ and SVDQuant](#import-awq-and-svdquant)
- [Storage and numerical behavior](#storage-and-numerical-behavior)
- [Runtime compatibility](#runtime-compatibility)

</details>

## Convert a model

Install the optional dependency from the repository root:

```bash
uv sync --locked --extra kitchen
```

Convert weights to FP8:

```bash
uv run unitunner weights convert models/model.safetensors models/model-fp8.safetensors \
  --precision float8_e4m3fn --device cuda
```

Export MXFP8, or dequantize an existing quantized file to BF16:

```bash
uv run unitunner weights convert models/model.safetensors models/model-mxfp8.safetensors \
  --precision mxfp8 --device cuda
uv run unitunner weights convert models/model-mxfp8.safetensors models/model-bf16.safetensors \
  --precision bf16 --device cuda
```

Inputs accept single files, numbered shards, safetensors index JSON files, Diffusers `transformer/` directories, and pipeline roots containing that directory. Place the output safetensors in ComfyUI `models/diffusion_models/` and use the corresponding model loader. Quantized files use Kitchen layouts; ordinary BF16 conversion needs no Kitchen dependency. Source and destination paths must differ.

```bash
uv run unitunner weights convert models/z_image_turbo_bf16.safetensors models/z_image_turbo_fp8.safetensors \
  --family z-image --precision float8_e4m3fn --device cuda
```

Training accepts ComfyUI files and Diffusers directories directly in `model.weights`. Released architectures have defaults; a local `config.json` can supply topology. Select semantic variants through `model.variant` and `model.version`. Supported families are Z-Image, FLUX, FLUX.2, FLUX3, MOSS, Qwen Image, Qwen Image 2.1, Krea2, Wan, and MiniMaxH3, using their published layouts.

Qwen Image 2.1 accepts ComfyUI weights with separate `gate_layer` / `proj` or fused `gate_up`, standard component prefixes, and indexed shards. Without topology metadata, tensor headers determine layer count, channels, attention heads, and MLP ratio. RoPE axis allocation and epsilon are not encoded in weights; custom values require `config.json` or explicit topology. Fused weights split in `[gate; up]` order. For INT8 ConvRot and other quantized files, only fused MLP input projections are decoded before splitting; other quantized projections retain their original storage. Loaded memory therefore exceeds the fully quantized file size. Saved `comfy_attention.config` entries are validated and filtered as inference-backend preferences; training uses its own attention settings. Missing weights, conflicting layouts, or inconsistent topology fail explicitly rather than initializing replacement parameters.

`weights import --kind transformer` also exports ComfyUI layout by default. Custom architectures can supply `--topology`; use `--output-format native` for native layout.

## Supported precisions

| `--precision` | Kitchen layout | Conversion |
| --- | --- | --- |
| `fp32`, `fp16`, `bf16` | Ordinary floating point | Cast floats; dequantize quantized inputs first |
| `float8_e4m3fn`, `float8_e5m2` | `TensorCoreFP8Layout` | Tensor-scaled FP8 |
| `mxfp8` | `TensorCoreMXFP8Layout` | E8M0 scaling per 32 elements |
| `nvfp4` | `TensorCoreNVFP4Layout` | FP8 scaling per 16 elements plus an FP32 global scale |
| `int8_tensorwise` | `TensorWiseINT8Layout` | Tensor-scaled INT8 |
| `int8_rowwise` | `TensorWiseINT8Layout` | INT8 scaled per output row |
| `int8_convrot` | `TensorWiseINT8Layout` | Row scaling with ConvRot |
| `convrot_w4a4`, `convrot_w4a8` | `TensorCoreConvRotW4A4Layout` | 4-bit weights with 4-bit or 8-bit activation layout |
| `asym_w4a8_int8` | `AsymW4A8Int8Layout` | Grouped 4-bit codebook weights |
| `w6a8_int8` | `AsymW4A8Int8Layout` | Grouped uniform 6-bit weights |
| `awq_w4a16` | `TensorCoreAWQW4A16Layout` | Import offline-calibrated weights |
| `svdquant_w4a4` | `TensorCoreSVDQuantW4A4Layout` | Import offline-calibrated weights and low-rank branches |
| `keep` | Existing layout | Validate and repackage, retaining quantized data |

W/A describe deployment weight and activation precision. Conversion stores weights and layout parameters; the consuming inference runtime performs activation quantization.

## Protect sensitive layers by model

The default `--family auto` prefers native topology metadata. Without metadata, family-specific key signatures drive detection. Set `--family` explicitly when the key structure is ambiguous. Names match training: `z-image`, `flux`, `flux2`, `flux3`, `moss-image`, `qwen-image`, `qwen-image-2-1`, `krea2`, `wan`, and `minimax-h3`.

```bash
uv run unitunner weights convert models/qwen21.safetensors models/qwen21-int8.safetensors \
  --family qwen-image-2-1 --precision int8_convrot --device cuda --on-incompatible keep
```

| Model | Default quantization targets | Main floating protections |
| --- | --- | --- |
| Z-Image | Main `layers` | Noise/context refiners, input/output, time/text embeddings, norms, modulation |
| FLUX/FLUX.2 | Double/single blocks | Input/output, time/guidance, norms, modulation |
| FLUX3 | Early content/text blocks, single blocks | Mode input/output, source/destination time, norms, modulation |
| MOSS Image | Main `layers` | Caption refiner, input/output, time, norms, AdaLN |
| Qwen Image/2.1 | `transformer_blocks` | Input/output, time/context projections, norms, modulation |
| Krea2 | Main `blocks` | Text fusion, input/output, time, norms, mod |
| Wan | Main `blocks` | Patch/text/time paths, output head, norms, modulation |
| MiniMaxH3 | Attention qkv/out, MLP fc1/fc2, block adaln linear | Token refiner, final output, input projections, norms |

All families exclude norms, biases, vocabularies, modulation, and control, motion, face, audio, and multimodal auxiliary branches, including branches nested inside core blocks. Krea2 retains `first`, `last`, `tmlp`, `txtfusion`, `txtmlp`, and `tproj`. During quantization, protected tensors keep their source dtype, including existing FP32 weights; BF16/FP16 sources cannot regain lost FP32 precision. Explicit ordinary floating conversion such as `--precision bf16` still casts selected floating tensors as requested.

These are conservative deployment policies, not quality-equivalence guarantees. `--include` narrows targets and cannot override protections. The `skipped` report includes protection and dimension reasons. `--family generic` selects two-dimensional weights for other models while retaining recognizable norm, embedding, time, and modulation paths; it cannot replace an architecture review. Known outer diffusion prefixes are stripped only for policy matching; `--output-format source` retains file keys.

### Text-encoder protections

Select `--family qwen2-5-vl|qwen3|qwen3-vl|qwen3-5|mistral|t5|umt5|clip` explicitly. These cover the project's Qwen2.5-VL, Qwen3, Qwen3-VL, Qwen3.5, Mistral, T5/UMT5, and CLIP encoders. Policies preserve the first and last block of each encoder stack using numeric layer indices, plus embeddings, LM heads, shared vocabularies, relative attention bias, normalization, complete vision/audio branches, multimodal projectors, and MTP. Qwen3.5 `in_proj_a`/`in_proj_b` remain floating too.

```bash
uv run unitunner weights convert models/text_encoder/model.safetensors models/text_encoder-fp8.safetensors \
  --family qwen3-vl --precision float8_e4m3fn --device cuda
```

Protection affects quantization selection; it never deletes decoder, LM-head, or other source keys. Stack boundaries use the entire shard inventory. Similar external language-model structures require an explicit family when `auto` cannot distinguish them. Conversion preserves existing tokenizer, configuration, and namespace requirements. Native text loaders accept every calibration-free layout for conditioning caches, previews, and inference, or quantize floating sources before encoding. See [text precision and quantization](training/text-encoding.md) for settings and family asset paths.

## Select layers

Within the family policy, quantization selects nonempty two-dimensional `*.weight` tensors and the root `weight` key. Biases and other tensors remain unchanged. This rule cannot distinguish linear weights from embeddings, so choose layers according to the actual model loader:

```bash
uv run unitunner weights convert models/model.safetensors models/model-int8.safetensors \
  --precision int8_convrot --device cuda --rotation-size 256 \
  --include 'transformer_blocks.*.weight' --exclude '*norm*' '*embed*' \
  --on-incompatible keep
```

Patterns are case-sensitive globs matched against full tensor keys; exclusions take precedence. Quote them to prevent shell expansion. Floating-point conversion casts all selected floating tensors and preserves integer buffers. Integer weights without layout metadata are rejected to avoid treating unknown packed bytes as ordinary weights.

| Option | Default | Meaning |
| --- | --- | --- |
| `--family` | `auto` | Family protection policy; includes `generic`, ten base families, and eight text-encoder families |
| `--device` | `cpu` | Conversion device, `cpu` or `cuda` |
| `--include` / `--exclude` | None | One or more tensor-key globs |
| `--rotation-size` | `256` | ConvRot group size; a power of four of at least 4 |
| `--group-size` | `16` | Grouped 4/6-bit quantization size; at least 4 and a divisor or multiple of 16 |
| `--on-incompatible` | `error` | Stop on unsupported layout dimensions; `keep` retains that layer in floating point and reports why |
| `--backend` | Automatic | Optional `eager`, `cuda`, or `triton`; accelerated backends require Kitchen's hardware and kernel support |
| `--layout-spec` | None | Layout descriptions for canonical offline-packed weights |

FP32 MXFP8/NVFP4 input uses Kitchen eager quantization on the selected device, preserving FP32 rounding when native kernels accept only FP16/BF16. INT8 ConvRot rotates weights and selects codes in FP32 even for BF16/FP16 sources. CPU conversion can use `--backend eager`, subject to the upstream method's shape and dtype constraints. `--on-incompatible keep` does not bypass missing calibration, corrupt metadata, or runtime kernel failures.

## Import AWQ and SVDQuant

Both upstream layouts explicitly require offline calibration in `quantize()`. Floating weights alone are insufficient. First obtain **Kitchen's canonical packed tensors** from the appropriate calibration pipeline, then provide their layout description. An AWQ input contains:

```text
transformer_blocks.0.attn.to_q.weight
transformer_blocks.0.attn.to_q.weight_scale
transformer_blocks.0.attn.to_q.weight_zeros
```

Save the following as `calibrated-layout.json`. Keys are module prefixes without `.weight`; use an empty string for a root weight. Dimensions and dtype must match the calibrated artifact:

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
uv run unitunner weights convert models/calibrated.safetensors models/model-awq.safetensors \
  --precision keep --layout-spec calibrated-layout.json
```

AWQ stores `weight` as int8 bytes of shape `[N, K/2]`. Low and high nibbles correspond to consecutive input columns. `weight_scale` and `weight_zeros` have shape `[K/group_size, N]`. Generic GPTQ/AWQ files may use different packing; renaming tensors or adding metadata does not make them compatible.

For SVDQuant, use `format: "svdquant_w4a4"` and include `weight_proj_down`, `weight_proj_up`, and `weight_smooth_factor`. Set parameters such as `act_unsigned` according to calibration. Kitchen's natural matrix and N-interleaved tiled storage are supported; runtime-only `mma_packed` or transposed storage must first be exported canonically. Files already containing `.comfy_quant` need no additional JSON. Requesting the same calibrated target format preserves its data; requesting either format from floating weights fails.

## Storage and numerical behavior

Each layer has a uint8 UTF-8 JSON tensor, `*.comfy_quant`, recording `format`, logical `orig_shape`, `orig_dtype`, and layout options. Layout serialization defines the auxiliary keys: FP8/INT8 use `weight_scale`; NVFP4 also uses `weight_scale_2`; grouped layouts retain `weight_s_rel`, `weight_s_channel`, and optional corrections and codebooks. AWQ/SVDQuant retain the calibration tensors described above. Metadata preserves INT8 row/rotation options and W4A8's `linear_dtype`.

MXFP8 E8M0 scales are stored as raw uint8 bytes for safetensors compatibility and reinterpreted as their original dtype on load. Block-scale ordering, padding, and bit packing follow Kitchen. Logical and padded storage shapes are retained separately, so dequantization restores the original dimensions for partial blocks. Older packed files lacking logical dimensions need a `--layout-spec`.

Excluded quantized layers and `keep` preserve auxiliary tensors, `input_scale`, and `pre_quant_scale`. Dequantization or requantization folds input smoothing from `pre_quant_scale` into weight columns and discards stale activation scales. Dequantization cannot recover information lost during quantization. SVDQuant float export uses upstream effective-weight reconstruction and cannot reproduce nonlinear activation quantization for arbitrary inputs.

Conversion reads tensors incrementally and spools output instead of holding the entire model in memory. A layer and its auxiliary tensors must still fit on the selected device. Output is built in temporary files; failures do not replace an existing destination. Temporary disk space is approximately twice the output size, in addition to existing files. A successful command prints JSON with converted and preserved counts and reasons for skipped layers. The file header records conversion options and the Kitchen version.

## Runtime compatibility

Native loaders across all ten base families support every calibration-free layout for adapter training, previews, and generation. Floating bases can also be quantized one layer at a time on GPU before training. All formats quantize weights only, reconstructing temporary floating weights for matrix multiplication without dynamic activation quantization; see [quantized-base training](training/quantized-base.md). Full parameter training needs floating weights. All artifacts must satisfy native loader key/topology requirements; use `weights import` on floating files before quantizing when needed.

Quantized outputs require consumers that support the corresponding Kitchen layouts. ComfyUI may quantize activations according to its layout, while UniTunner keeps activations floating; correct file interchange does not imply bit-identical outputs. The checked ComfyUI loader supports the directly quantizable formats above but has no AWQ/SVDQuant loading branches. Kitchen reload support for these two layouts does not imply current ComfyUI loading support. E5M2 files record `full_precision_matrix_mult: true`, selecting ComfyUI's dequantized floating matmul path because PyTorch does not support E5M2×E5M2 kernels. Actual inference also depends on the model loader, GPU, dtype, and kernels. MXFP8/NVFP4 execution needs an appropriate GPU path; this does not imply CPU execution support.

---

[Documentation](README.md) · [Parameter reference](training/parameters.md) · [Troubleshooting](troubleshooting.md)
