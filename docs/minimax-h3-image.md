# MiniMax-H3-Image

[English](en/minimax-h3-image.md) · [模型](training/models.md) · [配置示例](examples/README.md)

UniTuner 支持 `model.family = "minimax-h3-image"`、`model.variant = "base"` 的文生图、有序参考图编辑、流匹配 LoRA／全参数训练和一步 RGB 修复。图像去噪器、VAE 与 PDD 采样为原生实现；Qwen3-VL 文本编码器使用 `transformers`，不依赖 `diffusers`。H3 音视频与 H3-Image 的权重和缓存不可混用。

## 模型文件与显存

下载 `MiniMaxAI/MiniMax-H3-Image` 的原生 `transformer/`、`vae/`、`pdd_config.json`、`pdd_fused_heads.safetensors` 和 `processor/`，以及 `MiniMaxAI/MiniMax-H3` 的配套 `text_encoder/`。原生加载器不能使用 `*_diffusers/` 目录。官方 transformer 只有一个输出头，PDD 文件提供六步推理的输出头。

331 亿参数底模的 BF16 权重约占 61.74 GiB；rank-32 LoRA、梯度、Adam 状态和激活还要另计，32 GiB 显卡不能常驻训练。建议启用 `recompute = "block"`，从较低分辨率开始。`frozen-blocks` 将冻结块留在主内存，需要更多内存。缓存阶段加载文本编码器和图像 VAE；普通流匹配训练读取缓存，不再加载它们。修复训练另外加载图像 VAE。尚未使用完整 33B 权重完成端到端显存及输出一致性实测。

## 缓存与训练

下面的训练 TOML 可依次用于验证、缓存和训练；先替换全部路径。文生图不填写 `reference_sources`；编辑时每个参考目录与目标目录应有相同文件主体，目录顺序就是参考图顺序。同名 `.txt` 是原样使用的提示词或编辑指令，不进行提示词增强。

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

流匹配支持 REPA、LayerSync、TREAD，以及公共续训和预览工作流。一步修复需将数据设为 `kind = "restoration"`，提供 `low_quality_source` 或合成退化，并按[修复指南](training/restoration.md)配置 `[method.restoration]` 和 `[losses]`。支持 VAE 编码器适配器和 DINO／VGG／GAN 目标；流匹配专用的对齐目标不适用于修复。

## 生成与修复

`unituner generate` 使用独立配置，保留上述 `[model]`、`[model.minimax_h3]`、`[model.text.tokenizer]`，将下列 `output` 放在**文件开头、`[model]` 前**，再附加 `[sampling]`：

```toml
output = "/outputs/image.png"

[sampling]
prompt = "A coastal city at dawn"
seed = 42
# reference_images = ["/images/source.png"]
# width = 1024
# height = 1024
```

无参考图时默认输出 2048×2048；编辑默认继承首张参考图缩放后的长宽比。宽高均须为 32 的倍数，最多 16 张参考图。视觉参考图短边为 1024，VAE 参考图短边限制在 1024–2048。文本处理器按 `Picture 1: `、`Picture 2: ` 等顺序标记图片。固定 PDD 网格不接受 CFG、负面提示词或自定义步数。

`h3_size_policy = "official"` 在不指定尺寸时选择约 16 MP 面积；默认 `"comfyui"` 使用本地 2048×2048 条件尺寸，不影响参考图编码器。一步修复使用 `unituner restore`，指定 `input`、`output`、相同模型资产、对应的适配器及可选的编码器权重；训练权重可提供记录的修复 sigma，详见[修复推理](inference.md)。

适配器以 ComfyUI 格式导出；由于 H3-Image 的 Diffusers LoRA 投影映射尚未验证，不支持 Diffusers 格式发布。
