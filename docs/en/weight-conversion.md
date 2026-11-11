# Weight precision conversion

[简体中文](../weight-conversion.md) · English · [Documentation](README.md)

`unituner weights convert` changes safetensors weights to floating point or a supported quantized format. Use it to reduce weight storage, prepare a reusable quantized base, or export floating weights for compatible tools.

The default output uses ComfyUI denoiser format. Use `--output-format source` to retain the input format, especially when preparing weights for the same application. Match the output format to your intended loader.

<details>
<summary>On this page</summary>

- [Convert a model](#convert-a-model)
- [Supported precisions](#supported-precisions)
- [Protect sensitive layers by model](#protect-sensitive-layers-by-model)
- [Select layers](#select-layers)
- [Import AWQ and SVDQuant](#import-awq-and-svdquant)
- [Files, disk space, and results](#files-disk-space-and-results)
- [Runtime compatibility](#runtime-compatibility)

</details>

## Convert a model

Sync the default dependencies from the repository root:

```bash
uv sync --locked
```

Convert weights to FP8:

```bash
uv run unituner weights convert models/model.safetensors models/model-fp8.safetensors \
  --precision float8_e4m3fn --device cuda
```

Export MXFP8, or dequantize an existing quantized file to BF16:

```bash
uv run unituner weights convert models/model.safetensors models/model-mxfp8.safetensors \
  --precision mxfp8 --device cuda
uv run unituner weights convert models/model-mxfp8.safetensors models/model-bf16.safetensors \
  --precision bf16 --device cuda
```

Inputs accept single files, numbered shards, safetensors index JSON files, Diffusers `transformer/` directories, and pipeline roots containing that directory. Place the output safetensors in ComfyUI `models/diffusion_models/` and use the corresponding model loader. Quantized files use Kitchen layouts; ordinary BF16 conversion needs no Kitchen dependency. Source and destination paths must differ.

```bash
uv run unituner weights convert models/z_image_turbo_bf16.safetensors models/z_image_turbo_fp8.safetensors \
  --family z-image --precision float8_e4m3fn --device cuda
```

Training accepts ComfyUI files and Diffusers directories directly in `model.weights`. Released architectures have defaults; a local `config.json` can supply topology. Select semantic variants through `model.variant` and `model.version`. Supported families are Z-Image, FLUX, FLUX.2, FLUX3, MOSS, Qwen Image, Qwen Image 2.1, Krea2, Wan, and MiniMaxH3, using their published layouts.

Qwen Image 2.1 supports published ComfyUI files and indexed shards. Custom model structures need a matching `config.json`. Quantized files may use more memory when loaded than their on-disk size; allow extra working space.

`weights import --kind transformer` also exports ComfyUI layout by default. Custom architectures can supply `--topology`; use `--output-format native` for native layout.

## Supported precisions

| `--precision` | Conversion |
| --- | --- |
| `fp32`, `fp16`, `bf16` | Cast floats; dequantize quantized inputs first |
| `float8_e4m3fn`, `float8_e5m2` | Tensor-scaled FP8 |
| `mxfp8` | E8M0 scaling per 32 elements |
| `nvfp4` | FP8 scaling per 16 elements plus an FP32 global scale |
| `int8_tensorwise` | Tensor-scaled INT8 |
| `int8_rowwise` | INT8 scaled per output row |
| `int8_convrot` | Row scaling with ConvRot |
| `convrot_w4a4`, `convrot_w4a8` | 4-bit weights with 4-bit or 8-bit activation layout |
| `asym_w4a8_int8` | Grouped 4-bit codebook weights |
| `w6a8_int8` | Grouped uniform 6-bit weights |
| `awq_w4a16` | Import offline-calibrated weights |
| `svdquant_w4a4` | Import offline-calibrated weights and low-rank branches |
| `keep` | Validate and repackage, retaining quantized data |

W/A describe deployment weight and activation precision. Conversion stores weights and layout parameters; the consuming inference runtime performs activation quantization.

## Protect sensitive layers by model

The default `--family auto` prefers native topology metadata. Without metadata, family-specific key signatures drive detection. Set `--family` explicitly when the key structure is ambiguous. Names match training: `z-image`, `flux`, `flux2`, `flux3`, `moss-image`, `qwen-image`, `qwen-image-2-1`, `krea2`, `wan`, and `minimax-h3`.

```bash
uv run unituner weights convert models/qwen21.safetensors models/qwen21-int8.safetensors \
  --family qwen-image-2-1 --precision int8_convrot --device cuda --on-incompatible keep
```

Sensitive input/output, time/conditioning, normalization, and modulation layers automatically remain floating point. The conversion report lists protected or incompatible layers under `skipped`.

`--include` narrows the selected layers; it cannot override automatic protection. Use `--family generic` for models without a matching preset and check the converted result before use.

### Text-encoder protections

Select the matching preset with `--family qwen2-5-vl|qwen3|qwen3-vl|qwen3-5|mistral|t5|umt5|clip`. It preserves sensitive layers, embeddings, and multimodal components automatically.

```bash
uv run unituner weights convert models/text_encoder/model.safetensors models/text_encoder-fp8.safetensors \
  --family qwen3-vl --precision float8_e4m3fn --device cuda
```

Keep the model configuration, tokenizer, and processor assets alongside converted weights or configure their sources separately. For text caching and generation settings, see [text encoding](training/text-encoding.md).

## Select layers

Within the family policy, quantization selects nonempty two-dimensional `*.weight` tensors and the root `weight` key. Biases and other tensors remain unchanged. This rule cannot distinguish linear weights from embeddings, so choose layers according to the actual model loader:

```bash
uv run unituner weights convert models/model.safetensors models/model-int8.safetensors \
  --precision int8_convrot --device cuda --rotation-size 256 \
  --include 'transformer_blocks.*.weight' --exclude '*norm*' '*embed*' \
  --on-incompatible keep
```

Patterns are case-sensitive globs matched against full tensor keys; exclusions take precedence. Quote them to prevent shell expansion. Floating-point conversion casts all selected floating tensors and preserves integer buffers. Integer weights without layout metadata are rejected to avoid treating unknown packed bytes as ordinary weights.

| Option | Default | Meaning |
| --- | --- | --- |
| `--family` | `auto` | Family protection policy; includes `generic`, supported base models and text encoders |
| `--device` | `cpu` | Conversion device, `cpu` or `cuda` |
| `--include` / `--exclude` | None | One or more tensor-key globs |
| `--rotation-size` | `256` | ConvRot group size; a power of four of at least 4 |
| `--group-size` | `16` | Grouped 4/6-bit quantization size; at least 4 and a divisor or multiple of 16 |
| `--on-incompatible` | `error` | Stop on unsupported layout dimensions; `keep` retains that layer in floating point and reports why |
| `--backend` | Automatic | Optional `eager`, `cuda`, or `triton`; accelerated backends require Kitchen's hardware and kernel support |
| `--layout-spec` | None | Layout descriptions for canonical offline-packed weights |

CPU conversion can use `--backend eager`. Other backends require compatible hardware and dtypes. `--on-incompatible keep` skips shape-incompatible layers; it does not bypass missing calibration or invalid files.

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
uv run unituner weights convert models/calibrated.safetensors models/model-awq.safetensors \
  --precision keep --layout-spec calibrated-layout.json
```

AWQ stores `weight` as int8 bytes of shape `[N, K/2]`. Low and high nibbles correspond to consecutive input columns. `weight_scale` and `weight_zeros` have shape `[K/group_size, N]`. Generic GPTQ/AWQ files may use different packing; renaming tensors or adding metadata does not make them compatible.

For SVDQuant, use `format: "svdquant_w4a4"` and include `weight_proj_down`, `weight_proj_up`, and `weight_smooth_factor`. Set parameters such as `act_unsigned` according to calibration. Kitchen's natural matrix and N-interleaved tiled storage are supported; runtime-only `mma_packed` or transposed storage must first be exported canonically. Files already containing `.comfy_quant` need no additional JSON. Requesting the same calibrated target format preserves its data; requesting either format from floating weights fails.

## Files, disk space, and results

Use a different destination path to preserve the source. Allow temporary disk space of about twice the output size, in addition to existing files. A single layer and its associated data must fit on the selected device.

On success, the command prints JSON with converted counts, preserved quantized layers, and skip reasons. Review it before using the output. Dequantization cannot recover information lost during quantization. Keep all metadata when moving or sharing quantized files.

## Runtime compatibility

UniTuner supports adapter training, previews, and generation with the calibration-free formats above. Full-parameter training requires floating weights. See [quantized base training](training/quantized-base.md).

For other applications, confirm that the target loader supports both the model and quantization format. Low-bit weight storage alone does not guarantee acceleration or CPU support. Different runtimes may produce different outputs. AWQ/SVDQuant additionally require a loader that supports their calibrated format.


---

[Documentation](README.md) · [Parameter reference](training/parameters.md) · [Troubleshooting](troubleshooting.md)
