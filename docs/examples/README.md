# 配置示例

简体中文 · [English](../en/examples/README.md)

复制所需配置并填写模型与数据路径。相对路径以命令工作目录为基准。
训练方法见[训练指南](../training/README.md)，模型组合见[模型指南](../training/models.md)。

<details>
<summary>本页导航</summary>

- [如何使用示例](#如何使用示例)
- [完整教程配套配置](#完整教程配套配置)
- [训练与缓存](#训练与缓存)
- [图像修复](#图像修复)
- [推理](#推理)

</details>

## 如何使用示例

1. 复制到实验目录，先改模型、数据、缓存和输出路径。
2. 核对实际权重的 family/variant/version，阅读每个字段旁的中英注释。
3. 未注释的值组成该示例配方；注释掉的是可选项，不要全部打开。60 秒日志摘要、保留数量等默认值通常无需填写。
4. 训练按 validate→cache→train 运行；推理使用文件顶部标注的 generate、generate-video 或 restore。
5. 新实验用新目录/名称；只有继续同一训练时才使用 resume 状态。

填写步骤见[数据集](../training/datasets.md)、[训练流程](../training/README.md)和[参数手册](../training/parameters.md)。中英文文档共用同一组 TOML 配置。

## 完整教程配套配置

下列训练与推理配置成对使用，模型路径、产物名称和推理参数保持一致。

| 流程 | 训练配置 | 推理配置 | 教程 |
| --- | --- | --- | --- |
| Z-Image 风格 LoRA | [tutorial-lora.toml](tutorial-lora.toml) | [tutorial-generate.toml](tutorial-generate.toml) | [第一次训练](../getting-started.md) |
| Z-Image 一步修复 | [tutorial-restoration.toml](tutorial-restoration.toml) | [tutorial-restore.toml](tutorial-restore.toml) | [修复教程](../restoration-tutorial.md) |

## 训练与缓存

训练示例使用训练配置；纯 cache 示例只使用编码配置。通常先 `unituner cache CONFIG --device cuda`，
再 `unituner train CONFIG --device cuda`；仅有 denoiser 路径的训练配置需要先使用对应 cache 示例。

| 任务 | 示例 | 说明 |
| --- | --- | --- |
| 多组数据 | [data-groups.toml](data-groups.toml) | defaults 继承、独立缓存与 repeats |
| 纯图像缓存 | [image-cache.toml](image-cache.toml) | 只填写编码资产和数据 |
| 纯 VAE 校准 | [mid-timestep.toml](mid-timestep.toml) | 配对图像和 VAE，不需要 denoiser/缓存 |
| Z-Image adapter | [flow.toml](flow.toml) | 默认 LoRA，可把 algorithm 改为 loha／lokr |
| Z-Image full | [full.toml](full.toml) | 无 train.adapter，FP32 |
| 图像训练预览 | [flow-preview.toml](flow-preview.toml) | 训练时按提示词生成 |
| LayerSync＋TREAD＋REPA | [flow-features.toml](flow-features.toml) | SHA256 全零是占位符，必须换成编码器真实摘要；三项可独立删除 |
| FLUX Dev | [flux-flow.toml](flux-flow.toml) | T5 路径及本地 bundle 的 CLIP |
| FLUX Fill | [flux-fill-flow.toml](flux-fill-flow.toml) | 一个参考目录＋mask；改 Kontext 时更换权重／variant 并删除 mask_source |
| FLUX.2 | [flux2-flow.toml](flux2-flow.toml) | Klein4B base 无参考基线 |
| FLUX.2 有序参考 | [flux2-reference-flow.toml](flux2-reference-flow.toml) | 参考图与目标图使用相同 bucket |
| FLUX3 | [flux3-flow.toml](flux3-flow.toml) | 参考图规则见模型指南 |
| MOSS | [moss-flow.toml](moss-flow.toml) | 包含原生跳层预览参数，层号须匹配实际模型 |
| MOSS Edit | [moss-edit-flow.toml](moss-edit-flow.toml) | 参考图编辑；text-start 布局 |
| Qwen original | [qwen-image-flow.toml](qwen-image-flow.toml) | 普通文生图 |
| Qwen edit-2511 | [qwen-image-edit-flow.toml](qwen-image-edit-flow.toml) | 目标与参考按文件主体匹配 |
| Qwen layered | [qwen-image-layered-flow.toml](qwen-image-layered-flow.toml) | RGBA 前景层＋合成图 |
| Qwen Image 2.1 | [qwen-image-2-1-flow.toml](qwen-image-2-1-flow.toml) | 独立64通道vae与Qwen3-VL |
| Krea2 | [krea2-flow.toml](krea2-flow.toml) | RAW／Turbo 须匹配权重 |
| Wan 缓存 | [wan-cache.toml](wan-cache.toml)、[wan-single-frame-cache.toml](wan-single-frame-cache.toml) | 视频与独立条件帧缓存 |
| Wan 训练 | [wan-flow.toml](wan-flow.toml)、[wan-flow-preview.toml](wan-flow-preview.toml) | 不带／带视频预览 |
| H3 缓存 | [minimax-h3-cache.toml](minimax-h3-cache.toml) | 原生 AV JSONL，需完整音视频编码器 |
| H3 训练 | [minimax-h3-train.toml](minimax-h3-train.toml)、[minimax-h3-train-preview.toml](minimax-h3-train-preview.toml) | 已有缓存训练／带 AV 预览 |

H3 外部拆分缓存使用 [minimax-h3-cache-import.json](minimax-h3-cache-import.json)：
`unituner caches import SPECIFICATION DESTINATION`，这是显式转换入口，不是 TOML 训练配置。

## 图像修复

| 任务 | 示例 | 依赖与边界 |
| --- | --- | --- |
| Z-Image 默认损失配方 | [restoration.toml](restoration.toml) | DINOv3-ConvNeXt＋VGG19 权重 |
| Z-Image Turbo 油光肤色 | [oil-skin-lora.toml](oil-skin-lora.toml) | 真实配对图片，DINO/VGG/GAN 损失；需替换本地路径 |
| Z-Image full | [restoration-full.toml](restoration-full.toml) | 相同损失配方，denoiser 全参数，block 重算 |
| 合成退化 | [restoration-synthetic.toml](restoration-synthetic.toml) | 不提供成对 LQ；退化属于 dataset |
| 混合来源 | [restoration-mixed.toml](restoration-mixed.toml) | 真实配对与合成退化，每源独立缓存和 repeats |
| 独立／训练前 SNR 校准 | [restoration-calibration.toml](restoration-calibration.toml) | LRR＋L1，不依赖感知权重 |
| FLUX／FLUX.2／FLUX3 | [flux-restoration.toml](flux-restoration.toml)、[flux2-restoration.toml](flux2-restoration.toml)、[flux3-restoration.toml](flux3-restoration.toml) | LRR＋L1 配方 |
| MOSS | [moss-image-restoration.toml](moss-image-restoration.toml) | LRR＋L1 配方 |
| Qwen／Qwen2.1 | [qwen-image-restoration.toml](qwen-image-restoration.toml)、[qwen-image-2-1-restoration.toml](qwen-image-2-1-restoration.toml) | Qwen 仅 original |
| Krea2 | [krea2-restoration.toml](krea2-restoration.toml) | 原生训练后验 |

先替换路径，然后执行：

```bash
unituner mid-timestep docs/examples/restoration-calibration.toml --device cuda
unituner cache docs/examples/restoration-calibration.toml --device cuda
unituner train docs/examples/restoration-calibration.toml --device cuda
```

第一条不需要缓存，仅加载 VAE 和图像。它输出 JSON，不自动重写 TOML。
该示例的 calibrate=true 使训练前自动校准；若希望固定校准结果，关闭 calibrate 并填入输出的 sigma。
`midtimestep` 默认244，不表示所有模型均使用相同 sigma；完整公式与策略见[修复指南](../training/restoration.md)。

## 推理

这些配置采用独立 schema，不能交给 `unituner validate`（该命令当前只验证训练配置）。
使用对应推理命令会加载并验证配置，然后读取本地模型。

| 命令 | 示例 |
| --- | --- |
| `unituner generate CONFIG --device cuda` | [Z-Image](generate.toml)、[FLUX](flux-generate.toml)、[FLUX.2](flux2-generate.toml)、[FLUX3](flux3-generate.toml)、[MOSS](moss-generate.toml)、[MOSS Edit](moss-edit-generate.toml)、[Qwen](qwen-image-generate.toml)、[Qwen图层](qwen-image-layered-generate.toml)、[Krea2](krea2-generate.toml) |
| `unituner generate-video CONFIG --device cuda` | [Wan](wan-generate-video.toml)、[H3](minimax-h3-generate.toml) |
| `unituner restore CONFIG --device cuda` | [Z-Image](restore.toml)、[Qwen](qwen-image-restore.toml)、[Krea2](krea2-restore.toml) |

推理 adapter 的路径与 algorithm／rank／alpha 必须匹配训练产物；full 模型直接替换 model.weights。
修复推理的 encoder_weights 指向训练导出的 encoder adapter，并读取训练产物记录的 sigma。

---

[文档首页](../README.md) · [参数参考](../training/parameters.md) · [故障排查](../troubleshooting.md)
