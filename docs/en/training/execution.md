# Execution and GPU memory

[简体中文](../../training/execution.md) · English · [Documentation](../README.md)

GPU memory holds weights, optimizer state, gradients, and forward activations. Policies target different costs: recomputation saves activations, offload moves frozen weights or activations to the host, and BF16 AMP changes selected computation precision. Identify the bottleneck before choosing a policy.

<details>
<summary>On this page</summary>

- [Choose a policy](#choose-a-policy)
- [Defaults](#defaults)
- [Weight quantization before training](#weight-quantization-before-training)
- [Activation recomputation](#activation-recomputation)
- [Attention activation offload](#attention-activation-offload)
- [Frozen-block offload](#frozen-block-offload)
- [BF16 AMP and publication precision](#bf16-amp-and-publication-precision)
- [INT8 inference attention](#int8-inference-attention)
- [Combination and resume constraints](#combination-and-resume-constraints)
- [cuDNN and TF32 tuning](#cudnn-and-tf32-tuning)

</details>

> [!IMPORTANT]
> Residency is explicit: resident mode does not switch to offload after OOM. Automatic H2D capacity operates within the selected policy. `compile = "auto"` selects by device, not by free VRAM.

## Choose a policy

| Situation | Try first | Cost or constraint |
| --- | --- | --- |
| Weights and activations fit | Default resident, no recomputation | Baseline for throughput and memory |
| Weights fit but activations do not | `recompute = "block"` | Recomputes attention and feed-forward work; retains fewer activations |
| Attention-saved activations are large | `attention_activation_offload = true` | Host memory and D2H/H2D transfers |
| Frozen weights or transfers are too large | `quantization = "float8_e4m3fn"` or another supported layout | Smaller packed storage; temporary floating weights and dequantization overhead |
| Frozen adapter base weights do not fit | `residency = "frozen-blocks"` | Automatic shared cache, asynchronous prefetch, per-block recomputation |
| Flow matching can use mixed precision | `precision = "bf16"` | CUDA AMP; trainable parameters and optimizer masters remain FP32 |

Supported policies can be combined, without a promise of faster updates. Compare steady-state time and total GPU memory with the same model, dimensions, and batch size. Record first-call compilation separately.

## Defaults

These defaults usually need not be written explicitly:

```toml
[execution]
precision = "auto"
compile = false
residency = "resident"
recompute = "none"
recompute_interval = 1
attention_activation_offload = false
quantization = "none"
```

Compilation is off by default. Complete a run with defaults before trying `compile = true`, then compare steady-state speed. Initial startup and size changes can trigger compilation. `compile = "auto"` enables it on CUDA and leaves it off on CPU.

Restoration compilation can increase initial preparation time. Benefits depend on the model, resolution and device; compare complete-update time and peak GPU memory.

The following blocks are **edits** to existing sections. Merge them into your `[execution]` section rather than appending duplicate headers.

## Weight quantization before training

Add this to an adapter configuration whose base artifact is floating point:

```toml
[execution]
quantization = "float8_e4m3fn"
quantization_rotation_size = 256
quantization_group_size = 16
```

`quantization` selects weight storage, while `precision` selects floating compute precision. CUDA training quantizes one layer at a time on GPU, then reconstructs temporary weights for floating matmul. Rotation/group sizes apply to the relevant layouts only. See [quantized-base training](quantized-base.md) for formats and incompatible dimensions.

Native quantized artifacts load directly. `quantization = "none"` preserves their layout rather than reconstructing the entire model. Source artifacts and quantization settings participate in resume identity. Adapter `include_layers`/`exclude_layers` select trainable targets and cannot override quantization protections. Full-parameter training requires floating weights.

Text caches and inference use independent `[model.text]` precision/quantization settings and do not inherit `execution.quantization`. See [text encoding](text-encoding.md).

## Activation recomputation

Enable recomputation to save activation memory at the cost of additional computation:

```toml
[execution]
recompute = "block"
recompute_interval = 1
```

Resident mode can increase `recompute_interval` to checkpoint fewer DiT blocks: 1 selects every block, 2 roughly half, and 4 roughly a quarter. This interval does not affect VAE or DINO recomputation. Frozen-weight offload requires interval 1.

For restoration, keep VAE and DINO checkpointed while using spare memory to reduce DiT replay. Start by comparing complete-update time and peak memory with:

```toml
[execution]
residency = "resident"
compile = true
recompute = "block"
recompute_interval = 2
restoration_recompute = "block"
```

Try interval 4 if memory permits. Setting `recompute = "none"` then disables only DiT recomputation while explicit `restoration_recompute = "block"` keeps VAE and DINO checkpointed. The default, `restoration_recompute = "inherit"`, preserves the unified policy. This field applies only to restoration and does not affect flow-matching training. Keep your attention activation offload setting while comparing recomputation policies. Check headroom after a full backward and optimizer update at the largest bucket. This policy does not automatically adjust intervals based on free memory or retry after OOM.

## Attention activation offload

```toml
[execution]
attention_activation_offload = true
```

Temporarily store some attention activations on CPU and bring them back when needed. This requires CUDA and increases host-memory use and transfer overhead.

Combine it with recomputation or frozen-weight offload when needed. Check host memory and training speed to avoid moving the bottleneck to CPU RAM or transfers.

## Frozen-block offload

```toml
[execution]
residency = "frozen-blocks"
```

This policy supports CUDA adapters only. Frozen block weights remain on CPU and transfer for forward and backward recomputation. Adapters and embeddings/output layers outside blocks remain on the compute device; it is not whole-model CPU offload.

This mode requires `recompute = "block"` and `recompute_interval = 1`; omitted recomputation settings are filled in automatically. GPU cache capacity adjusts automatically, so start without a manual limit. Verify that the largest training dimensions complete an update; automatic adjustment does not guarantee that arbitrary larger sizes fit.

Usually omit `frozen_weight_cache_mb` and `use_pinned_memory_for_block_swap`. Set the former in MiB only to limit additional retained frozen weights; zero keeps space for the current block. This is not a total VRAM limit: activations and optimizers need memory too. If host memory is tight, set the latter to `false`, accepting potentially slower transfers.


## BF16 AMP and publication precision

```toml
[execution]
precision = "bf16"
```

BF16/FP16 supports CUDA flow matching and restoration. See [precision](precision.md) for defaults, FP16 overflow handling and restoration feature precision.

To reduce published file size independently, use `output.save_precision = "bf16"` or `"fp16"`. Publication dtype and training computation precision are separate; resume state retains training dtypes.

## INT8 inference attention

Install Kitchen, then explicitly select it in generation configuration:

```bash
uv sync --locked
```

```toml
[sampling]
attention_backend = "kitchen-int8"
```

Image previews set the same field in `[[preview.samples]]`; standalone `restore` puts it at the root. Wan/H3 use native sampling sections. Only denoiser inference changes; VAE and text encoding retain their original paths.

The default is `"torch"`. `kitchen-int8` is for inference only and requires CUDA and supported model shapes. Select `"torch"` when unsupported. Changing attention backends can change outputs.

## Combination and resume constraints

`trainable-blocks` is unsupported. FP16 uses AMP with FP32 master parameters and fixed loss scaling. Calibration-free weight layouts support frozen-base LoRA/LoHa/LoKr training with temporary weight reconstruction and floating compute; see [quantized-base training](quantized-base.md). Full training uses resident weights. Frozen blocks support adapters only and require whole-block recomputation.

Changing precision, quantization or compilation settings can make old resume states incompatible. Compare settings in new output directories with the same model, data and seed. A fixed seed does not guarantee identical results across different devices or settings.


## cuDNN and TF32 tuning

CUDA convolutions use these runtime defaults, which can be overridden:

```toml
[runtime]
cudnn_benchmark = true
cudnn_deterministic = false
```

cudnn_enabled defaults to true. cuda_allow_tf32 defaults to false and can be enabled explicitly for supported FP32 matrix operations; it does not turn BF16 into TF32. Benchmarking searches kernels at startup and can cost more when bucket shapes vary. For repeatability choose cudnn_deterministic=true and normally keep benchmarking off. These are backend controls, not train/optimizer settings, and can be used by cache or inference too.

Change one policy at a time: record baseline update time and peak memory, enable block recomputation if activations do not fit, then compare weight offload/quantization when frozen weights dominate. Enable compilation after eager execution works reliably. Detailed fields and constraints: [parameters](parameters.md).

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
