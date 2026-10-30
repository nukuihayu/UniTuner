# Text precision and quantization

[简体中文](../../training/text-encoding.md) · English · [Training](README.md)

Text encoders accept floating or Kitchen packed weights and can quantize floating sources before conditioning caches or inference. Text settings are independent of the denoiser: `model.text` controls encoding; `execution.quantization` controls the training base.

<details>
<summary>On this page</summary>

- [Local single-file weights](#local-single-file-weights)
- [Tokenizer sources](#tokenizer-sources)
- [Quantize before encoding](#quantize-before-encoding)
- [Load a converted encoder](#load-a-converted-encoder)
- [Safeguards and cache identity](#safeguards-and-cache-identity)

</details>

## Local single-file weights

Every model family accepts a local single-file `model.text_encoder`. Safetensors shards accept any standard numbered shard and require the complete set. Existing directories remain supported. Supported family namespaces are detected from weight keys, with strict key/shape/quantization validation.

Single-file topology resolution prefers a matching `.json` sidecar, then sibling `config.json`, then a bundled released default. Defaults cover Z-Image Qwen3-4B, FLUX T5-XXL/CLIP-L, FLUX.2 Dev/Klein, FLUX3/Krea2 Qwen3-VL-4B, Qwen Image and Qwen Image 2.1. Custom variants must supply their matching JSON. MOSS Qwen3.5 requires a sidecar or sibling configuration; Wan retains its native configuration; MiniMaxH3 retains `model.minimax_h3.text_config`.

Model topology never downloads at runtime. Tokenizer/processor sources remain independent; where no Hub default exists, a single file's parent directory is the local fallback. Set `model.text.tokenizer.source` for separate assets. Cache identity includes every shard and the effective topology JSON.

## Tokenizer sources

Tokenizers/processors default to Musubi's Hub sources. Downloads are restricted to vocabulary, template and processor metadata files; encoder weights remain local under `model.text_encoder`, with their local model configuration. Remote Python code is never executed.

| Model | Repository / subfolder |
| --- | --- |
| Z-Image | `Tongyi-MAI/Z-Image` / `tokenizer` |
| FLUX | T5: `google/t5-v1_1-xxl`; CLIP: `openai/clip-vit-large-patch14` |
| FLUX.2 Dev | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` |
| FLUX.2 Klein 4B / 9B | `Qwen/Qwen3-4B` / `Qwen/Qwen3-8B` respectively |
| Qwen Image | `Qwen/Qwen-Image` / `tokenizer`; processor: `Qwen/Qwen-Image-Edit` / `processor` |
| Krea2 | `Qwen/Qwen3-VL-4B-Instruct` |
| Wan | `Wan-AI/Wan2.1-T2V-14B` / `google/umt5-xxl` |
| MiniMaxH3 | `MiniMaxAI/MiniMax-H3` / `processor` |

```toml
[model.text.tokenizer]
# source = "/models/my-tokenizer" # Local directory or Hub repository ID
# subfolder = "tokenizer"        # Explicit sources default to their root
revision = "main"                # Prefer a commit SHA for reproducibility
offline = false
```

Omit `source` to use the defaults above, or `revision` to use the repository default branch. `offline = true` uses existing Hub cache files only; local directories never access the network. Gated repositories use your Hugging Face credentials. CLI overrides such as `--set model.text.tokenizer.offline=true` use the same schema. `HF_HUB_OFFLINE=1` also disables Hub access.

FLUX accepts a separate `[model.text.clip_tokenizer]` table; Qwen Image accepts `[model.text.processor]`, both with the same fields. When omitted, these sources inherit the primary offline flag; set revisions separately. An explicit Qwen Image primary `source` also supplies its processor unless the processor table is provided, so that directory must contain complete processor assets. Other multimodal families use the primary tokenizer table for their complete processor.

MOSS Image, FLUX3 and Qwen Image 2.1 retain their local text directory defaults and accept explicit overrides. Wan selects vocabulary through the tokenizer table. MiniMaxH3 `text_processor` is optional. Cache identity includes actual vocabulary, template and processor contents: rerun cache after changing them.

## Quantize before encoding

Install the optional dependency and add this fragment to your existing configuration:

```bash
uv sync --locked --extra kitchen
```

```toml
[model.text]
precision = "bf16"
quantization = "float8_e4m3fn"
```

Keep `model.text_encoder` pointing to local source assets. `unituner cache train.toml --device cuda` loads the floating text model on CPU, quantizes each projection on GPU, returns packed layers to CPU, then places the encoder on its inference device. The complete floating encoder never needs to occupy GPU memory; host memory must still fit the floating source. CPU execution quantizes on CPU. Source files are unchanged.

| Field | Default | Purpose |
| --- | --- | --- |
| `precision` | Omitted | Floating computation dtype: `fp32`, `fp16`, or `bf16`; omission retains the caller default, usually FP32 in CLI caching and generation |
| `quantization` | `none` | Target storage for floating sources; `none` preserves a packed artifact's layout |
| `quantization_rotation_size` | `256` | ConvRot group size, a power of four of at least 4; layer dimensions must be compatible |
| `quantization_group_size` | `16` | Group size for grouped 4/6-bit layouts; layout and layer constraints apply |
| `max_length` | Family default | Token budget only where the native interface supports it; quantization does not change family limits |

Supported layouts are `float8_e4m3fn`, `float8_e5m2`, `mxfp8`, `nvfp4`, `int8_tensorwise`, `int8_rowwise`, `int8_convrot`, `convrot_w4a4`, `convrot_w4a8`, `asym_w4a8_int8`, and `w6a8_int8`. `fp8` and `convrot-int8` alias E4M3 and INT8 ConvRot. AWQ/SVDQuant need calibration and are not accepted by the text runtime.

Every layout quantizes weights only. Inputs stay floating; linear operations reconstruct temporary floating weights for matmul. A4/A8 names do not enable activation quantization. Low-bit storage does not imply low-bit matmul acceleration, and each active layer still needs temporary floating memory. Incompatible dimensions fail explicitly. Use standalone conversion with `--on-incompatible keep` when individual layers must remain floating.

The same settings apply to text loading for training previews and standalone generation, using each workflow's text device. Denoiser, VAE, and text precision remain independent. To change only floating text computation, set `precision` and retain `quantization = "none"`.

## Load a converted encoder

Prepare matching model configuration in the output directory, then convert weights. Tokenizer, chat-template, and processor assets can come from the independent sources described below:

```bash
uv run unituner weights convert models/qwen3/model.safetensors models/qwen3-fp8/model.safetensors \
  --family qwen3 --precision float8_e4m3fn --device cuda
```

Set `model.text_encoder` to `models/qwen3-fp8` and keep `model.text.quantization = "none"`. Per-layer `*.comfy_quant` metadata selects layouts while scales, codebooks, and rotation tensors retain their storage dtypes. An explicit quantization target must match existing packed layers; lossy weights are never implicitly requantized. Conversion preserves source keys/prefixes, so automatic detection continues to work; an explicit `model.text_source_format` override must match the original namespace.

File conversion also supports `--precision bf16`, `fp16`, and `fp32`, reconstructing quantized input first. This cannot recover lost information. Online quantization after selecting a computation dtype can produce different codes from offline quantization in the source dtype. For a direct comparison, first convert offline sources to the same floating dtype and use matching quantization parameters, device, and backend.

| Model family | Text conversion preset |
| --- | --- |
| Z-Image, FLUX.2 Klein | `qwen3` |
| MOSS Image | `qwen3-5` |
| FLUX | `t5` for the primary text directory and `clip` for the auxiliary directory; convert separately |
| FLUX.2 Dev | `mistral` |
| FLUX3, Qwen Image 2.1, Krea2, MiniMaxH3 | `qwen3-vl` |
| Qwen Image | `qwen2-5-vl` |
| Wan | `umt5`; retain native topology metadata and separate tokenizer assets |

FLUX online settings apply to both T5 and CLIP. Convert each offline for different layouts, retaining the existing auxiliary asset structure. Sharded conversion merges output; do not leave old shards or stale indexes in the destination directory.

MiniMaxH3 uses language layers 0–49. Online preparation protects layers 0 and 49 automatically. When converting a complete, untruncated Qwen3-VL source offline, also pass `--exclude '*layers.49.*'`; a file matching the native truncated topology is preferable. Text loaders accept artifacts with already quantized first/last layers or other protected paths and preserve their packed storage without requiring reconversion. Keys, shapes, layouts, and explicit quantization parameters must still pass validation.

## Safeguards and cache identity

Online preparation and file conversion share [text safeguards](../weight-conversion.md#text-encoder-protections): numeric first/last blocks, normalization, embeddings, vocabularies/LM heads, vision/audio branches, multimodal projectors, MTP, and Qwen3.5 sensitive projections. Hidden-state selection, padding, image/video tokens, and RoPE retain native family conventions. T5 activation dtype checks use floating computation precision instead of packed storage dtype.

Cache identity includes source assets, explicit precision/quantization settings, and effective layouts/computation dtypes. Rerun `cache` after changing them. Feature storage follows existing schemas, usually FP32, while retaining FLUX3's native BF16 feature boundary. Storing low-precision results in FP32 does not recover information lost during computation.

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
