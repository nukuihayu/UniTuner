# Architecture

[简体中文](../architecture.md) · **English**

[Documentation](README.md) · [Project home](../../README.md)

UniTuner separates native model computation, training methods, execution policies, and external artifacts. Models share orchestration while retaining native conditioning, time units, prediction direction, and weight semantics.

Training and output schemas live in `config/train.py` and `config/output.py`; `config/training.py` composes them. `training_validation.py` checks run policy and previews once, then validates each dataset group. Floating precision names are compute-free contracts in `artifacts/formats/precision.py`, with runtime dtypes and contexts in `precision.py`. Image model loading lives in `models/image.py`; training bindings and publication live in `training/integrations/image_models.py` and `image_publication.py`. Shared previews import only the selected family when needed, keeping text providers optional for restoration previews.

<details>
<summary>On this page</summary>

- [Layout and ownership](#layout-and-ownership)
- [Adapter parameters and computation](#adapter-parameters-and-computation)
- [Native model ownership](#native-model-ownership)
- [Training and inference boundaries](#training-and-inference-boundaries)
- [Configuration, caches, and artifacts](#configuration-caches-and-artifacts)
- [Weight quantization ownership](#weight-quantization-ownership)
- [Resource lifecycle](#resource-lifecycle)
- [Validation](#validation)

</details>

## Layout and ownership

Paths are relative to the repository root:

```text
src/unituner/
├── cli/             # Argument parsing, configuration overrides, lazy dispatch
├── config/          # Strict shared schemas and combination validation
├── models/          # Native model families and specialized components
├── data/            # Media, cache contracts, datasets, and degradation
├── adapters/        # LoRA / LoHa / LoKr mathematics and installation
├── training/        # Objectives, parameter plans, updates, and previews
├── execution/       # Compilation, recomputation, attention, and residency
├── checkpoints/     # Update-boundary state saving and restoration
├── artifacts/       # Weight conversion, validation, and atomic publication
└── inference/       # Shared image generation and one-step restoration

tests/
├── config/          # Configuration boundaries and combinations
├── models/          # Native formulas, layouts, and gradients
├── comparison/      # Optional allocator test entry
└── support/         # Shared test assets and assertions

docs/
├── README.md        # Chinese documentation entry
├── getting-started.md
├── architecture.md
├── training/        # Chinese training and model guides
├── en/              # Matching English guides and example index
└── examples/        # Shared executable configurations and Chinese index
```

Tests are also grouped by adapters, artifacts, checkpoints, data, execution, training, and other responsibilities.

## Adapter parameters and computation

`adapters/lora.py`, `loha.py`, and `lokr.py` define trainable parameters, scaling, and weight export.
`construction.py` builds adapters, `attachment.py` attaches and removes them, and `loading.py` and `encoder.py` handle artifacts.
Optimized computation lives in `adapters/ops/`, consistently named by algorithm:

| Files | Responsibility |
| --- | --- |
| `ops/lora.py` / `ops/lora_cuda.py` | LoRA linear computation and fused residual scaling/addition |
| `ops/loha.py` / `ops/loha_cuda.py` | Fused LoHa Hadamard products and scaling |
| `ops/lokr.py` / `ops/lokr_cuda.py` | Fused dense LoKr Kronecker construction and backward products |

Each ordinary module owns eligibility checks, autograd, and native PyTorch fallbacks; its `*_cuda.py` module owns lazily loaded Triton kernels.
Operation modules do not depend on adapter parameter classes or attachment orchestration. Optimizations preserve matrix multiplications, low-precision rounding, and gradient reduction order.
Unsupported layouts, compilation, or automatic differentiation cases retain native expressions.
Matching numerical tests live in `tests/adapters/ops/test_{lora,loha,lokr}.py`; parameter formula and loading tests remain in `tests/adapters/`.

## Native model ownership

`models/<family>/` owns the transformer, VAE, layers, positional geometry, text encoding, loading, and genuinely family-specific configuration/cache/sampling components. Components can be explicitly shared without adding empty wrappers for directory symmetry. Native models do not import training workflows or fall back to external reference trainers.

`models/components/transformer_artifacts.py` resolves precedence among explicit configuration, embedded metadata, and external configuration, and rejects topology conflicts. Family loaders own model construction, weight layout conversion, and precision policy.

Shared configuration belongs in `config/`: model identity, parameter mode, data, execution, and output. Only independent native semantics belong in `models/<family>/config/`. References, masks, controls, and text budgets use shared fields first, followed by family capability checks. A shared field does not imply every value is supported by every model; silently ignoring it is not compatibility.

## Training and inference boundaries

| Owner | Responsibility | Boundary |
| --- | --- | --- |
| `training/integrations/` | Bind native conditions, time, targets, and eligible adapter projections | Does not duplicate the training loop |
| `training/methods/` | Formulas and state for flow matching, restoration, and feature methods | Does not guess native time or silently convert inputs |
| `training/workflows/` | Image/Wan/H3 updates, optimizer, accumulation, saving, and previews | Does not define native network layers |
| `execution/` | Install policies on declared native regions | Does not rewrite objectives or silently downgrade configuration |
| `inference/` and native inference components | Sampling, VAE decoding, and publication loading | Does not depend on the training loop for generation |

Adapters/full select trainable parameters. Flow matching/restoration select the objective. LayerSync, REPA, and TREAD are composable method components; execution is another dimension. Supported combinations are validated at input boundaries rather than through scattered model-name checks or late patches.

`training/objectives.py` owns LayerSync, REPA, and TREAD installation and cleanup through one entry shared by image, Wan, and H3 workflows. Execution policies remain owned by `execution/`.

## Configuration, caches, and artifacts

The CLI loads immutable Pydantic configuration and rejects unknown fields. Loading configuration does not create models or probe GPUs. Media/text optional dependencies are imported at application boundaries. Relative paths use the command's working directory.

`training/output.py` owns run directory name sanitization, timestamps, and collision handling. `training/telemetry.py` uses those rules or an already resolved run name, and owns metric recording and the event writer lifecycle.

`data/cache/` owns producer/consumer content identity, manifests, and tensor records. Identity includes relevant model and numerical inputs. Rebuild after changing media, captions, or conditions. Crops, posteriors, and condition encoding preserve native supervision semantics.

`data/cache/storage.py` owns record collision checks, atomic writes, and file digests. Family storage entry points validate native tensor and metadata contracts.

`artifacts/` owns explicit external weight-key conversion, strict tensor contracts, and atomic publication. Algorithm/consumer prefix conversion stays at this boundary without changing internal training weights. `checkpoints/` owns resumable model/method state, optimizer, RNG, and data position. Inference publications and training checkpoints have separate contracts.

## Weight quantization ownership

`artifacts/formats/precision_policy.py` defines shared base-model and text-encoder safeguards for standalone conversion and training preparation. `artifacts/quantization.py` delegates packing and serialization to upstream layouts; `quantized_loading.py` validates logical dimensions, sidecars, and native model keys. `models/components/weight_quantization.py` prepares each layer, and `quantized.py` owns the common quantized linear computation.

Initialization loads the floating base on CPU, quantizes one layer at a time on the selected GPU and returns its packed representation to CPU, applies residency placement, then installs adapters, optimizers, and execution owners. CPU training quantizes on CPU. Replacing frozen projections therefore precedes live optimizer parameters and never requires the complete floating base on GPU.

All calibration-free layouts quantize weights only. Forward reconstructs temporary floating weights for matmul; backward reconstructs them from saved packed storage for input gradients. Scales/corrections follow weight residency while preserving their own dtypes. No complete floating base is retained, and INT8 dynamic activation quantization is not enabled. Quantization settings participate in resume identity; see [quantized-base training](training/quantized-base.md).

## Resource lifecycle

A training session explicitly owns adapters, execution installations, method modules, and optimizers. Resources are registered after successful creation and released in reverse order on completion or error, after related backward/device work finishes.

`models/components/forward_override.py` owns order checks and restoration for instance `forward` overrides. Adapter and execution installations compose this object while retaining responsibility for device transfers, weights, and graph resources.

Recomputation, compilation, and residency install through named native regions and must preserve live optimizer Parameter identities. Frozen weights can still be read for input gradients, so residency and VBAR leases cover those reads. Process-level VBAR initialization happens before Torch import; see [execution](training/execution.md).

Feature capture covers only the current forward, without duplicate capture during recomputation or graph retention across updates. Previews isolate RNG and do not advance the training cursor. Checkpoints are saved at completed update boundaries; restore validates model, configuration, and cache identity before restoring live state.

## Validation

Tests cover model formulas, gradients, weight formats, training updates and resource cleanup.


---

[Documentation](README.md) · [Parameter reference](training/parameters.md) · [Troubleshooting](troubleshooting.md)
