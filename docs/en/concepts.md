# Understand the training workflow

[简体中文](../concepts.md) · English · [Documentation](README.md)

Configuration answers three questions: what should the model learn, which parameters can change, and how should computation fit into device memory? Keeping these choices separate makes it easier to adapt a run to another model or task.

<details>
<summary>On this page</summary>

- [How an image reaches the model](#how-an-image-reaches-the-model)
- [What flow matching learns](#what-flow-matching-learns)
- [What LoRA and full training change](#what-lora-and-full-training-change)
- [Why caching comes first](#why-caching-comes-first)
- [How one-step restoration differs](#how-one-step-restoration-differs)
- [Count updates and samples](#count-updates-and-samples)
- [What memory policies change](#what-memory-policies-change)
- [Publications and training state](#publications-and-training-state)

</details>

## How an image reaches the model

Image generation models commonly work in a compressed **latent** space rather than full pixel space. A VAE encodes images into latents. A text encoder supplies prompt features. The denoiser predicts a direction in the generation process under those conditions, and the VAE decodes the resulting latent into an image.

These components have different responsibilities. A denoiser cannot necessarily use another model's VAE or text encoder. Families differ in latent channels, normalization, text features, and condition layouts, so assets must match the selected family and variant.

## What flow matching learns

For the common noise-minus-data convention, the training process can be illustrated as:

```text
z₀ = encode(image)
ε  = Gaussian noise
zσ = (1 − σ) · z₀ + σ · ε
v* = ε − z₀
loss = mean((model(zσ, time(σ), text) − v*)²)
```

`σ` is the noise fraction. Sampling images, noise, and time teaches a direction from noise toward data. At generation time, a sampler follows that direction over multiple time steps.

This is an explanatory formula. Native implementations own timestep units, shifted schedules, prediction signs, and conditioning. Z-Image's sign, for example, is handled at its model boundary. An integer timestep is not a universal noise fraction across families.

## What LoRA and full training change

LoRA freezes the original weight `W` and trains a low-rank branch:

```text
W_effective = W + (alpha / rank) · B · A
```

`rank` sets the intermediate dimension and `alpha/rank` scales the update. Higher rank increases adapter capacity, parameters, gradients, and optimizer state. Whether it helps requires evaluation on your data. Freezing the base removes its parameter gradients and optimizer state, but backward still traverses base computations to obtain input gradients. LoRA can therefore still use substantial GPU memory.

LoHa and LoKr represent weight updates with different structures, selected through `train.adapter.algorithm`. Full mode directly updates selected denoiser parameters, usually requires more training state, and publishes a complete model.

These choices determine what can change. They can serve flow matching or restoration; `method.kind` selects the objective.

## Why caching comes first

Ordinary image flow matching does not update the VAE or text encoder. Precomputing latents, text features, and conditions avoids repeating their work on every update:

1. `validate` checks fields and supported combinations.
2. `cache` reads media and encoding components to produce training representations.
3. `train` reads the cache and denoiser and performs gradient updates.

Caches bind to actual inputs. Rebuild after changing images, captions, references, encoding assets, or crop rules. Changing only learning rate or update count does not change encoded inputs.

Restoration needs live LQ encoding and pixel supervision, so it still reads LQ/HQ images during training. It is not an entirely offline latent workflow. Previews also encode prompts and decode outputs, requiring their VAE and text assets.

## How one-step restoration differs

Restoration starts from an actual degraded image: `LQ → encode → fixed-time prediction → decode → compare with HQ`. It does not add Gaussian noise to the LQ input. For noise-minus-data models, the restoration step is approximately `z_clean = z_lq − σ · velocity`.

`σ` selects the restoration time, which can be calibrated from data. An encoder adapter helps move LQ representations toward a useful restoration distribution. Pixel losses, perceptual losses, and LRR supervise different aspects of the result. See the [restoration tutorial](restoration-tutorial.md).

## Count updates and samples

`max_updates` counts optimizer updates; `accumulation_steps` counts microbatches before an update. On one device:

```text
batch_size = 2
accumulation_steps = 4
Each update processes 2 × 4 = 8 samples
max_updates = 1000 → 8000 samples processed, including repeated sampling
```

Accumulation aggregates more samples per update without placing a larger microbatch in memory at once. It does not remove a single sample's activation cost or guarantee the same speed/numerics as a direct batch of equal size. Warmup, saving, and preview intervals are measured in updates.

## What memory policies change

Recomputation spends additional forward work to save fewer activations. Offload moves weights or activations to host memory. BF16 AMP lowers the precision of selected forward computations. Each addresses a different memory cost, so choose based on the measured bottleneck.

These settings preserve the intended task but affect runtime and floating-point computation. Some belong to resume identity; changing a memory policy does not necessarily preserve the checkpoint's numerical trajectory. See [execution policies](training/execution.md).

Weight quantization reduces frozen-base storage and transfers. Floating sources can be quantized one layer at a time on GPU before training, or converted into reusable artifacts. Computation temporarily reconstructs each layer and uses floating inputs for forward and input gradients. This adds dequantization work and still needs temporary floating-weight memory. Quantization supports frozen-base adapter training; full mode requires floating weights. See [quantized-base training](training/quantized-base.md).

## Publications and training state

A publication contains the model or adapter needed for inference. A checkpoint also retains optimizer state, RNG, and data position so optimization can resume.

Restoration can additionally publish an encoder adapter and record effective sigma. Use denoiser and encoder publications from the same run with a consistent time setting. See [inference and model use](inference.md).

---

[Documentation](README.md) · [Parameter reference](training/parameters.md) · [Troubleshooting](troubleshooting.md)
