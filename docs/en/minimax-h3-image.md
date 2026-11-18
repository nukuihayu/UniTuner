# MiniMax-H3-Image

[简体中文](../minimax-h3-image.md) · [Models](training/models.md) · [Examples](examples/README.md)

UniTuner supports text-to-image, ordered reference-image editing, flow-matching LoRA/full training, and one-step RGB restoration for `model.family = "minimax-h3-image"` and `model.variant = "base"`. The denoiser, image VAE, and PDD sampler are native; the Qwen3-VL text encoder uses `transformers`, not `diffusers`. H3 audio/video weights and caches are not interchangeable with H3-Image.

## Assets and memory

Download the original `transformer/`, `vae/`, `pdd_config.json`, `pdd_fused_heads.safetensors`, and `processor/` from `MiniMaxAI/MiniMax-H3-Image`, plus the matching `text_encoder/` from `MiniMaxAI/MiniMax-H3`. Do not use the `*_diffusers/` directories with native loaders. The published transformer has one output head; the PDD head file supplies the six-step inference heads.

The 33B denoiser alone occupies about 61.74 GiB in BF16. Rank-32 LoRA, gradients and Adam states need additional memory, as do activations; a 32 GiB GPU cannot hold the resident BF16 base. Use `recompute = "block"` and start at a modest resolution. Frozen-block residency moves frozen block weights to CPU but needs substantially more host RAM. The text encoder and image VAE are used during caching; ordinary flow training reads their cached outputs. Restoration training also loads the image VAE. No full-size 33B end-to-end memory or output-parity measurement has been completed.

## Cache and train

Use one training TOML for validation, caching, and training; replace all paths. Omit `reference_sources` for text-to-image. Each reference directory must contain the same image stems as the target directory, in the order to be used for edits. Captions (`.txt` files) contain the exact prompts or editing instructions; no prompt expansion occurs.

```toml
[model]
family = "minimax-h3-image"
variant = "base"
weights = "/models/MiniMax-H3-Image/transformer"
vae = "/models/MiniMax-H3-Image/vae"
text_encoder = "/models/MiniMax-H3/text_encoder"
pdd_config = "/models/MiniMax-H3-Image/pdd_config.json"
pdd_heads = "/models/MiniMax-H3-Image/pdd_fused_heads.safetensors"

[model.minimax_h3]
text_config = "/models/MiniMax-H3/text_encoder"
text_device = "cpu"

[model.text.tokenizer]
source = "/models/MiniMax-H3-Image/processor"
offline = true

[dataset.groups.main]
kind = "image"
source = "/data/targets"
cache_directory = "/data/h3-image-cache"
resolution = [768, 768]
batch_size = 1
# reference_sources = ["/data/references"]

[train]
max_updates = 1000

[train.adapter]
algorithm = "lora"
rank = 32
alpha = 32.0

[execution]
precision = "bf16"
recompute = "block"

[output]
directory = "/runs/h3-image"
```

```bash
uv run unituner validate h3-image.toml --for cache
uv run unituner cache h3-image.toml --device cuda
uv run unituner validate h3-image.toml --for train
uv run unituner train h3-image.toml --device cuda
```

Flow training supports REPA, LayerSync, TREAD, and the shared checkpoint/preview workflow. For one-step restoration, change the dataset to `kind = "restoration"`, supply `low_quality_source` or synthetic degradation, and configure `[method.restoration]` and `[losses]` as in the [restoration guide](training/restoration.md). The image VAE encoder adapter and DINO/VGG/GAN objectives are available; restoration does not use the flow-only alignment objectives.

## Generate and restore

For `unituner generate`, use a separate configuration with root-level `output` **before** the same `[model]`, `[model.minimax_h3]`, and `[model.text.tokenizer]` sections, then add `[sampling]`:

```toml
output = "/outputs/image.png"

[sampling]
prompt = "A coastal city at dawn"
seed = 42
# reference_images = ["/images/source.png"]
# width = 1024
# height = 1024
```

The default text-only size is 2048×2048; editing follows the first reference image's scaled aspect ratio. Both dimensions must be multiples of 32. Up to 16 reference images are supported. Vision references use a 1024-pixel short edge; VAE references use a 1024–2048-pixel short edge. The text processor labels images `Picture 1: `, `Picture 2: `, and so on. The fixed PDD schedule runs without CFG, negative prompts, or a configurable step count.

`h3_size_policy = "official"` selects an approximately 16 MP area when no dimensions are specified; the default `"comfyui"` policy follows the local 2048×2048 conditioning size. Neither changes the reference-image encoders. To run one-step restoration, use `unituner restore` with `input`, `output`, the same model assets, a matching published adapter and optional encoder weights. The recorded restoration sigma can be read from trained weights; see [restoration inference](inference.md).

Adapters export in ComfyUI format. Diffusers LoRA publication is not supported because its H3-Image projection mapping is unverified.
