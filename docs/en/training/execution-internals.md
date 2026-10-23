# Execution mechanics and resource lifetimes

[简体中文](../../training/execution-internals.md) · [Documentation](../README.md)

Use [execution policies](execution.md) to configure training. This page explains computation, weight residency, and backward-read boundaries for understanding resources and maintaining model implementations.

<details>
<summary>On this page</summary>

- [One training update](#one-training-update)
- [Compilation and recomputation](#compilation-and-recomputation)
- [Frozen-weight ownership](#frozen-weight-ownership)
- [H2D cache](#h2d-cache)
- [VBAR mapped weights](#vbar-mapped-weights)
- [Attention activation offload](#attention-activation-offload)
- [Installation and cleanup](#installation-and-cleanup)

</details>

## One training update

```text
Load and prepare base → Attach adapters/select parameters → Install execution
                                                               ↓
Read microbatch → Forward and loss → Backward/accumulation → Optimizer update
                                                               ↓
                                            Save when due → Preview → Next update
```

Policies install through model-owned regions while preserving parameter identities, state keys, and optimizer storage. Native position encoding, time units, conditioning layouts, and prediction formulas remain model responsibilities.

## Compilation and recomputation

`compile = false` uses eager execution; `true` requires CUDA Inductor; `"auto"` enables it on CUDA and uses eager on CPU. Only gradient-enabled training block computation is compiled. Previews and standalone generation use native evaluation.

Prefetch, page allocation, and resource cleanup stay outside the block compilation boundary. Compilation may change floating-point rounding; bitwise equality with eager is not guaranteed. See [precision](precision.md) for defaults and configuration.

Block recomputation is non-reentrant and preserves CPU/CUDA RNG needed for replay. Resident mode can select blocks through `recompute_interval`; H2D/VBAR require every block to replay, with interval 1. There is no separate FFN recomputation option.

## Frozen-weight ownership

Frozen weights need no parameter gradients, but input gradients can still read them during backward. Storage acquired for a forward must remain valid until the graph finishes its last read.

| Object | Responsibility |
| --- | --- |
| CPU source | Immutable weight contents for the lifetime of the installation |
| Weight lease | Binds concrete storage validity to forward/backward reads |
| Prefetcher | Prepares eligible later blocks on a copy stream |
| H2D cache / VBAR arena | Manages ordinary CUDA storage or mapped-page residency |
| Installation | Attaches native block execution and restores resources/callables after backward |

Adapters, trainable parameters, and necessary non-block model components stay on the compute device. Weight offload does not manage the entire model, all activations, or optimizer state.

## H2D cache

Frozen-block mode groups contiguous frozen parameters by dtype into CPU payloads. Computation uses native-shaped views; noncontiguous tensors retain strides and are handled separately. CUDA streams/events coordinate transfers, and active leases cannot be evicted or overwritten.

With no explicit `frozen_weight_cache_mb`, capacity is calibrated from observed backward peaks, free memory, and temporary conversion needs. An explicit value limits retained storage plus one current-block workspace; zero leaves that workspace. It is not a process-wide GPU memory cap.

When all weights cannot remain resident, selected complete regions stay cached while others stream on demand. Capacity changes trigger replanning, and active graph reads take precedence over budgets. An unseen larger bucket may still require smaller input dimensions.

## VBAR mapped weights

VBAR uses AIMDO to manage virtual addresses and physical GPU pages. Immutable CPU sources remain available; contents are staged and transferred on demand. When pages are unavailable, protected fallback storage is used. Page pins and fallback storage obey the same weight-lease lifetimes.

The training CLI initializes the allocator before importing Torch. Cleanup completes outstanding device work and backward reads before releasing mappings, staging, and installations. An inference forward's end is not equivalent to a training graph's last backward read.

H2D and VBAR are explicitly selected policies, without automatic switching between them. Both support CUDA adapter training on frozen bases. Trainable-block swapping is unsupported.

## Attention activation offload

Saved-tensor hooks stage SDPA's saved non-leaf CUDA activations while preserving shape and stride, and restore them on backward reads. This manages different objects from weight offload and can be combined within supported configurations.

Checkpoint replay avoids retaining the same offloaded activations again. Training still needs differentiable attention; inference-only INT8 attention cannot supply training input gradients directly.

## Installation and cleanup

`TrainingSession` registers successfully constructed resources immediately and closes them in reverse order on completion or error. `ForwardOverride` checks instance-forward ownership and restores the predecessor. Later owners close first, repeated close is harmless, and class lookup is restored when no instance override originally existed.

Compilation, recomputation, and adapters do not discover or guess other wrappers at runtime. Frozen-weight installation checks close order before releasing transfer/graph resources, restoring original devices, and relinquishing its forward override.

Save resume state only at completed update boundaries after clearing gradients. Frozen-reference snapshots omit duplicate frozen weights but require identical source weights when restored. See [saving and resuming](README.md#saving-and-resuming).

---

[Documentation](../README.md) · [Parameter reference](parameters.md) · [Troubleshooting](../troubleshooting.md)
