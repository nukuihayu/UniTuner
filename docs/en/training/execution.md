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
- [VBAR residency](#vbar-residency)
- [BF16 AMP and publication precision](#bf16-amp-and-publication-precision)
- [INT8 inference attention](#int8-inference-attention)
- [Combination and resume constraints](#combination-and-resume-constraints)
- [cuDNN and TF32 tuning](#cudnn-and-tf32-tuning)

</details>

> [!IMPORTANT]
> Residency is explicit: resident mode does not switch to offload after OOM. Automatic H2D capacity and VBAR page management operate within the selected policy. `compile = "auto"` selects by device, not by free VRAM.

## Choose a policy

| Situation | Try first | Cost or constraint |
| --- | --- | --- |
| Weights and activations fit | Default resident, no recomputation | Baseline for throughput and memory |
| Weights fit but activations do not | `recompute = "block"` | Recomputes attention and feed-forward work; retains fewer activations |
| Attention-saved activations are large | `attention_activation_offload = true` | Host memory and D2H/H2D transfers |
| Frozen weights or transfers are too large | `quantization = "float8_e4m3fn"` or another supported layout | Smaller packed storage; temporary floating weights and dequantization overhead |
| Frozen adapter base weights do not fit | `residency = "frozen-blocks"` | Automatic shared cache, asynchronous prefetch, per-block recomputation |
| Manage frozen weights through virtual memory | `residency = "vbar"` | Physical residency follows device capacity; NVIDIA CUDA and vbar extra |
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

The default uses eager execution without compilation startup cost. Use `compile` to compare compiled training. Explicit true enables CUDA Inductor; `"auto"` selects Inductor on CUDA and eager on CPU. Compilation covers training computations. Previews and standalone inference use native evaluation paths. Enable compilation after comparing steady-state throughput; compilation can change floating-point rounding and update trajectories.

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

Use non-reentrant whole-block recomputation, preserving RNG state while replaying attention and feed-forward computations during backward:

```toml
[execution]
recompute = "block"
recompute_interval = 1
```

Resident mode can increase `recompute_interval` to recompute selected blocks; frozen-weight offload requires interval 1.

## Attention activation offload

```toml
[execution]
attention_activation_offload = true
```

On CUDA, non-leaf tensors saved by SDPA for backward move to pinned CPU memory and return on demand. This does not offload the entire attention module or optimizer state.

It supports resident none/block, frozen blocks, and supported VBAR recomputation. Previews disable these hooks. Host memory and transfer bandwidth can become the new bottleneck.

## Frozen-block offload

```toml
[execution]
residency = "frozen-blocks"
```

This policy supports CUDA adapters only. Frozen block weights remain on CPU and transfer for forward and backward recomputation. Adapters and embeddings/output layers outside blocks remain on the compute device; it is not whole-model CPU offload.

Frozen-block offload uses CPU master weights, asynchronous prefetch and an automatically sized GPU cache. Omitted recompute is inferred as block; explicit none is rejected. Leave cache capacity and pinned-memory options omitted initially. Compare stable transfer/compute time and host RAM use; this policy is for frozen adapter bases, not full training. See [execution internals](execution-internals.md) for cache allocation details.

Usually omit `frozen_weight_cache_mb` and `use_pinned_memory_for_block_swap`. An explicit MiB value caps retained storage plus one largest frozen block for current computation. Zero permits one block and cannot also hold an equal-sized lookahead block. This does not cap total VRAM, active-graph storage, activations, optimizer state, or casts. Pinned masters consume non-pageable host memory; explicit false disables them.

Buffers are reused only after all graph readers finish and their CUDA events are respected. CPU masters remain immutable. Block recomputation is required; full training is unsupported. Outstanding graphs may temporarily exceed the cache budget rather than overwrite live weights.

## VBAR residency

Install the dependency:

```bash
uv sync --locked --extra media --extra text --extra vbar
```

Then edit the configuration:

```toml
[execution]
residency = "vbar"
```

VBAR uses comfy-aimdo to manage virtual frozen-weight residency. It reuses resident pages, transfers faults from immutable CPU sources, and automatically prefetches for forward/backward. Unlike frozen-block caching, it does not use `frozen_weight_cache_mb`.

Prefetch checks available memory. If actual allocator retries increase, completed-backward feedback increases the prefetch reserve to reduce failed speculative allocations, without a manual threshold. Demand allocations still fail when they cannot fit. Compare H2D and VBAR at actual training dimensions: throughput depends on computation, transfers, and residency.

It requires NVIDIA CUDA adapter training and supports compilation and BF16 AMP. The CLI initializes AIMDO before Torch import; Python callers must preserve that ordering.

VBAR pages are outside Torch `max_memory_allocated`. Also inspect `ExecutionInstallation.vbar_statistics().loaded_bytes`. For host memory, consider both `pinned_staging_bytes` and `registered_source_bytes`. Lower Torch counters alone do not establish lower total GPU memory use.

## BF16 AMP and publication precision

```toml
[execution]
precision = "bf16"
```

BF16/FP16 AMP supports CUDA flow matching and image restoration. See [precision configuration](precision.md) for automatic detection, weight storage, FP16 loss scaling and default BF16 VGG/DINO. Trainable master parameters remain FP32; backward and optimizer updates run outside forward autocast.

To reduce published file size independently, use `output.save_precision = "bf16"` or `"fp16"`. Publication dtype and training computation precision are separate; resume state retains training dtypes.

## INT8 inference attention

Install Kitchen, then explicitly select it in generation configuration:

```bash
uv sync --locked --extra media --extra text --extra kitchen
```

```toml
[sampling]
attention_backend = "kitchen-int8"
```

Image previews set the same field in `[[preview.samples]]`; standalone `restore` puts it at the root. Wan/H3 use native sampling sections. Only denoiser inference changes; VAE and text encoding retain their original paths.

Default `"torch"` uses native SDPA. Kitchen INT8 lacks the backward path required here and fails when explicitly called with gradients. CPU and unsupported head shapes/layouts do not silently fall back. INT8 may produce different outputs from Torch.

## Combination and resume constraints

`trainable-blocks` is unsupported. FP16 uses AMP with FP32 master parameters and fixed loss scaling. Calibration-free weight layouts support frozen-base LoRA/LoHa/LoKr training with temporary weight reconstruction and floating compute; see [quantized-base training](quantized-base.md). Full training uses resident weights. Frozen blocks/VBAR support adapters only; Both VBAR and frozen blocks require whole-block recomputation.

Compilation, fused LoRA computation, and selected precision boundaries belong to resume identity. Compare policies in separate output directories starting from the same base. Do not assume a changed policy preserves a checkpoint's numerical trajectory. A fixed seed alone also does not guarantee byte-identical CUDA results.

See [execution internals](execution-internals.md) for transfer streams, weight leases, VBAR staging, and compile boundaries.

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
